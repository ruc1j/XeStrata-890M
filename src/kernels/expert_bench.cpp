// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/expert_bench.cpp - the routed experts' decode time on this GPU (XeStrata-890M): native_expert_grouped on
// the IQ2_XS model's shapes (n_embd 2560, n_ff 640; gate/up IQ2_S or IQ2_XXS, down Q2_0) for a verify window of T
// tokens of 10 experts each, the blobs drawn from a pool in host USM as the expert cache holds them.  Random weights
// from a fixed seed: `--save F` writes the outputs, `--check F` compares a changed kernel's to them.
//     expert_bench [--save F | --check F] [gu_type]
#include "strata/core/gpu.hpp"
#include "strata/kernels/iq_kernels.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {
uint16_t f16_bits(float f) {   // normal values only
    uint32_t u;
    std::memcpy(&u, &f, 4);
    const uint32_t s = (u >> 16) & 0x8000, e = ((u >> 23) & 0xff) - 127 + 15, m = (u >> 13) & 0x3ff;
    return (uint16_t) (s | (e << 10) | m);
}
}  // namespace

int main(int argc, char** argv) {
    std::string save, check;
    int gu = 22;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--save") && i + 1 < argc) save = argv[++i];
        else if (!std::strcmp(argv[i], "--check") && i + 1 < argc) check = argv[++i];
        else gu = std::atoi(argv[i]);
    }
    const int64_t n_embd = 2560, n_ff = 640;
    const int d = 42;
    if (!strata::kernels::native_expert_supported(gu, d, n_embd, n_ff)) { std::printf("unsupported %d/%d\n", gu, d); return 1; }
    const auto L = strata::kernels::native_expert_layout(gu, d, n_embd, n_ff);
    const size_t blob = (L.bytes + 255) / 256 * 256;
    const size_t pool_n = (size_t) (4ull << 30) / blob;
    uint8_t* pool = static_cast<uint8_t*>(strata::gpu::alloc_host(pool_n * blob));
    if (!pool) { std::printf("host USM pool failed\n"); return 1; }
    std::mt19937_64 rng(77);
    for (size_t i = 0; i < pool_n * blob / 8; ++i) reinterpret_cast<uint64_t*>(pool)[i] = rng();
    // every block's fp16 scale normal: the gate/up blocks (IQ2 families keep d first) and the Q2_0 down blocks
    const size_t gu_bsz = strata::kernels::iq_row_bytes(gu, 256), d_bsz = strata::kernels::iq_row_bytes(d, 64);
    for (size_t b = 0; b < pool_n; ++b) {
        uint8_t* p = pool + b * blob;
        for (size_t o = 0; o + gu_bsz <= L.down_off; o += gu_bsz) {
            const uint16_t h = f16_bits(0.0005f + 0.00001f * (float) (rng() % 100));
            std::memcpy(p + o, &h, 2);
        }
        for (size_t o = L.down_off; o + d_bsz <= L.bytes; o += d_bsz) {
            const uint16_t h = f16_bits(0.001f + 0.00001f * (float) (rng() % 100));
            std::memcpy(p + o, &h, 2);
        }
    }
    void* s = strata::gpu::stream_create();
    std::vector<float> all;
    for (int T = 1; T <= 6; ++T) {
        // T tokens of 10 distinct experts each; the groups are the distinct experts, the entries their tokens
        std::vector<std::vector<int>> by_expert;
        std::vector<size_t> ids;
        for (int t = 0; t < T; ++t) {
            std::vector<size_t> mine;
            while (mine.size() < 10) {
                // a token's experts: some shared with the window's earlier tokens, as routing is
                const size_t e = (!ids.empty() && rng() % 3 == 0) ? ids[rng() % ids.size()] : rng() % pool_n;
                if (std::find(mine.begin(), mine.end(), e) == mine.end()) mine.push_back(e);
            }
            for (size_t e : mine) {
                auto it = std::find(ids.begin(), ids.end(), e);
                if (it == ids.end()) { ids.push_back(e); by_expert.push_back({t}); }
                else by_expert[(size_t) (it - ids.begin())].push_back(t);
            }
        }
        const int G = (int) ids.size();
        std::vector<unsigned long long> gptr(G);
        std::vector<int32_t> gstart(G + 1, 0), ent_tok, ent_dst;
        for (int g = 0; g < G; ++g) {
            // EXPERT_BENCH_SAME=1: every group reads the same blob (cached): what the kernels cost without the memory
            gptr[g] = (unsigned long long) (pool + (std::getenv("EXPERT_BENCH_SAME") ? 0 : ids[g]) * blob);
            for (int t : by_expert[g]) { ent_tok.push_back(t); ent_dst.push_back((int32_t) ent_dst.size()); }
            gstart[g + 1] = (int32_t) ent_tok.size();
        }
        const int E = (int) ent_tok.size();
        std::vector<uint8_t> xq((size_t) T * (n_embd / 32) * 36);
        for (auto& b : xq) b = (uint8_t) rng();
        for (size_t i = 0; i < (size_t) T * (n_embd / 32); ++i) {
            const uint16_t dd = f16_bits(0.02f), sum = f16_bits(0.5f);
            std::memcpy(&xq[i * 36], &dd, 2);
            std::memcpy(&xq[i * 36 + 2], &sum, 2);
        }
        unsigned long long* d_gptr; int32_t *d_gstart, *d_ng, *d_tok, *d_dst; void *d_x, *d_scr; float* d_out;
        strata::gpu::alloc_device(&d_gptr, G * 8);
        strata::gpu::alloc_device(&d_gstart, (G + 1) * 4);
        strata::gpu::alloc_device(&d_ng, 4);
        strata::gpu::alloc_device(&d_tok, E * 4);
        strata::gpu::alloc_device(&d_dst, E * 4);
        strata::gpu::alloc_device(&d_x, xq.size());
        strata::gpu::alloc_device(&d_scr, strata::kernels::native_expert_scratch_bytes(E, n_ff));
        strata::gpu::alloc_device(&d_out, (size_t) E * n_embd * 4);
        strata::gpu::copy(d_gptr, gptr.data(), G * 8);
        strata::gpu::copy(d_gstart, gstart.data(), (G + 1) * 4);
        strata::gpu::copy(d_ng, &G, 4);
        strata::gpu::copy(d_tok, ent_tok.data(), E * 4);
        strata::gpu::copy(d_dst, ent_dst.data(), E * 4);
        strata::gpu::copy(d_x, xq.data(), xq.size());
        auto run = [&] {
            strata::kernels::native_expert_grouped(L, d_gptr, d_gstart, d_ng, d_dst, d_tok, G, E, d_x, d_scr, d_out, s);
        };
        run();
        strata::gpu::stream_sync(s);
        std::vector<float> y((size_t) E * n_embd);
        strata::gpu::copy(y.data(), d_out, y.size() * 4);
        all.insert(all.end(), y.begin(), y.end());
        const int reps = 300;
        const auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; ++r) run();
        strata::gpu::stream_sync(s);
        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / reps;
        std::printf("T %d  experts %2d  entries %2d  %7.1f us  %5.1f GB/s (each blob read once)\n", T, G, E, us,
                    (double) G * L.bytes / us / 1e3);
        for (void* p : {(void*) d_gptr, (void*) d_gstart, (void*) d_ng, (void*) d_tok, (void*) d_dst, d_x, d_scr,
                        (void*) d_out})
            strata::gpu::free(p);
    }
    if (!save.empty()) {
        FILE* f = std::fopen(save.c_str(), "wb");
        std::fwrite(all.data(), 4, all.size(), f);
        std::fclose(f);
        std::printf("saved %zu outputs\n", all.size());
    }
    if (!check.empty()) {
        std::vector<float> ref(all.size());
        FILE* f = std::fopen(check.c_str(), "rb");
        const size_t n = f ? std::fread(ref.data(), 4, ref.size(), f) : 0;
        if (f) std::fclose(f);
        if (n != all.size()) { std::printf("check: %zu outputs in the file, want %zu\n", n, all.size()); return 1; }
        size_t diff = 0;
        double worst = 0;
        for (size_t i = 0; i < n; ++i)
            if (std::memcmp(&ref[i], &all[i], 4) != 0) {
                ++diff;
                worst = std::max(worst, std::fabs((double) ref[i] - all[i]) / (std::fabs((double) ref[i]) + 1e-6));
            }
        std::printf("check: %zu of %zu outputs differ, largest relative difference %.3g\n", diff, n, worst);
        return diff == 0 ? 0 : 2;
    }
    strata::gpu::free(pool);
    return 0;
}
