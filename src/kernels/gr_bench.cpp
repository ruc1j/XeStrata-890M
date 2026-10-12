// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/gr_bench.cpp - the hyper-connection read's time on this GPU (XeStrata-890M): fused_gr_read_multi with the
// Q8_0 down and up projections (STRATA_HC_Q8's) for T = 1..6 tokens, as a verify window calls it for every layer.
// Random values from a fixed seed: `--save F` writes the outputs (mixed, R, inject), `--check F` compares to them.
//     gr_bench [--save F | --check F] [--layers N]   (N matrices round robin, default 8, so the L2 does not hold them)
#include "strata/core/gpu.hpp"
#include "strata/kernels/fused_gr.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#include <thread>

namespace {
constexpr int N = 2560, HC = 4, D = N * HC, LR = 320;
uint16_t f16_bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    const uint32_t s = (u >> 16) & 0x8000, e = ((u >> 23) & 0xff) - 127 + 15, m = (u >> 13) & 0x3ff;
    return (uint16_t) (s | (e << 10) | m);
}
// one matrix in kNativeQ8Rows' layout: each row's int8 values, then its fp16 scales
std::vector<uint8_t> q8_rows(int rows, int cols, std::mt19937& rng) {
    const size_t row = (size_t) cols / 32 * 34;
    std::vector<uint8_t> m((size_t) rows * row);
    for (int r = 0; r < rows; ++r) {
        uint8_t* p = m.data() + (size_t) r * row;
        for (int c = 0; c < cols; ++c) p[c] = (uint8_t) (int8_t) ((int) (rng() % 255) - 127);
        for (int b = 0; b < cols / 32; ++b) {
            const uint16_t d = f16_bits(0.0001f + 0.00001f * (float) (rng() % 50));
            std::memcpy(p + cols + 2 * b, &d, 2);
        }
    }
    return m;
}
template<class T> T* up(const std::vector<T>& v) {
    T* d = nullptr;
    strata::gpu::alloc_device(&d, v.size() * sizeof(T));
    strata::gpu::copy(d, v.data(), v.size() * sizeof(T));
    return d;
}
}  // namespace

int main(int argc, char** argv) {
    std::string save, check;
    int layers = 8;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--save") && i + 1 < argc) save = argv[++i];
        else if (!std::strcmp(argv[i], "--check") && i + 1 < argc) check = argv[++i];
        else if (!std::strcmp(argv[i], "--layers") && i + 1 < argc) layers = std::atoi(argv[++i]);
    }
    std::mt19937 rng(5);
    std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
    std::vector<uint8_t*> qd(layers), qu(layers);
    std::vector<uint16_t*> wd(layers), wu(layers), wi(layers);
    std::vector<float*> wn(layers);
    for (int l = 0; l < layers; ++l) {
        qd[l] = up(q8_rows(LR, D, rng));
        qu[l] = up(q8_rows(D, LR, rng));
        std::vector<uint16_t> bf((size_t) LR * D);
        for (auto& x : bf) x = (uint16_t) (0x3c00 | (rng() & 0x7f));   // small BF16 values
        wd[l] = up(bf);
        wu[l] = up(bf);
        std::vector<uint16_t> inj((size_t) HC * D);
        for (auto& x : inj) x = (uint16_t) (0x3c00 | (rng() & 0x7f));
        wi[l] = up(inj);
        std::vector<float> norm(D);
        for (auto& x : norm) x = 1.0f + 0.1f * uni(rng);
        wn[l] = up(norm);
    }
    constexpr int TMAX = 6;
    std::vector<float*> R(TMAX), bo(TMAX), ip(TMAX), lo(TMAX), rs(TMAX), injo(TMAX), mixed(TMAX);
    for (int t = 0; t < TMAX; ++t) {
        std::vector<float> r(D), b(N), i(HC);
        for (auto& x : r) x = uni(rng);
        for (auto& x : b) x = 0.1f * uni(rng);
        for (auto& x : i) x = uni(rng);
        R[t] = up(r);
        bo[t] = up(b);
        ip[t] = up(i);
        strata::gpu::alloc_device(&lo[t], LR * 4);
        strata::gpu::alloc_device(&rs[t], HC * 4);
        strata::gpu::alloc_device(&injo[t], HC * 4);
        strata::gpu::alloc_device(&mixed[t], N * 4);
    }
    float* scratch = nullptr;
    strata::gpu::alloc_device(&scratch, (size_t) strata::kernels::kFusedGrMaxT * D * 4);
    void* s = strata::gpu::stream_create();
    std::vector<float> all;
    for (int T = 1; T <= TMAX; ++T) {
        auto args = [&](int l) {
            std::vector<strata::kernels::FusedGrArgs> a(T);
            for (int t = 0; t < T; ++t) {
                auto& x = a[t];
                x.R = R[t]; x.R_out = nullptr; x.apply = false; x.bo_prev = bo[t]; x.inj_prev = ip[t];
                x.w_norm = wn[l]; x.w_down = wd[l]; x.w_up = wu[l]; x.w_inject = wi[l];
                x.lo = lo[t]; x.rs = rs[t]; x.inject_out = injo[t]; x.mixed = mixed[t];
                x.q8_down = qd[l]; x.q8_up = qu[l];
            }
            return a;
        };
        const auto a0 = args(0);
        strata::kernels::fused_gr_read_multi(a0.data(), T, scratch, s);
        strata::gpu::stream_sync(s);
        for (int t = 0; t < T; ++t) {
            std::vector<float> y(N), inj(HC);
            strata::gpu::copy(y.data(), mixed[t], N * 4);
            strata::gpu::copy(inj.data(), injo[t], HC * 4);
            all.insert(all.end(), y.begin(), y.end());
            all.insert(all.end(), inj.begin(), inj.end());
        }
        std::vector<std::vector<strata::kernels::FusedGrArgs>> per(layers);
        for (int l = 0; l < layers; ++l) per[l] = args(l);
        const int reps = 400;
        const auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; ++r) strata::kernels::fused_gr_read_multi(per[r % layers].data(), T, scratch, s);
        strata::gpu::stream_sync(s);
        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / reps;
        const double bytes = 2.0 * LR * D / 32 * 34;
        std::printf("T %d  %7.1f us  %5.1f GB/s (the two Q8_0 matrices)\n", T, us, bytes / us / 1e3);
    }
    if (std::getenv("BENCH_LINGER") != nullptr) std::this_thread::sleep_for(std::chrono::seconds(3));   // a tracer's flush
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
        for (size_t i = 0; i < n; ++i) diff += std::memcmp(&ref[i], &all[i], 4) != 0;
        std::printf("check: %zu of %zu outputs differ\n", diff, n);
        return diff == 0 ? 0 : 2;
    }
    return 0;
}
