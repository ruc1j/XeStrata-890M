// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/mmvq_bench.cpp - the dense decode products' time on this GPU: native_iq4_xs_mmvq on the model's shapes
// (2560 -> 10240 attn_qkv, 2560 -> 6144 attn_gate, 2560 -> 640 the shared expert) for 1..8 columns, with the bytes
// read per call and the rate.  Random weights from a fixed seed: `mmvq_bench --save F` writes every output, and
// `mmvq_bench --check F` compares a changed kernel's outputs to them (bitwise, then the largest relative difference).
#include "strata/core/gpu.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {
uint16_t f16_bits(float f) {   // normal values only (the benchmark's scales)
    uint32_t u;
    std::memcpy(&u, &f, 4);
    const uint32_t s = (u >> 16) & 0x8000, e = ((u >> 23) & 0xff) - 127 + 15, m = (u >> 13) & 0x3ff;
    return (uint16_t) (s | (e << 10) | m);
}
struct Shape { int n_in, n_out; const char* name; };
}  // namespace

int main(int argc, char** argv) {
    const std::string save = argc == 3 && std::strcmp(argv[1], "--save") == 0 ? argv[2] : "";
    const std::string check = argc == 3 && std::strcmp(argv[1], "--check") == 0 ? argv[2] : "";
    const Shape shapes[] = {{2560, 10240, "qkv"}, {2560, 6144, "gate"}, {2560, 640, "shexp"}, {6144, 2560, "ssm_out"}};
    std::mt19937 rng(1234);
    void* s = strata::gpu::stream_create();
    std::vector<float> all;
    for (const Shape& sh : shapes) {
        const int blocks = sh.n_in / 256;
        const size_t wbytes = (size_t) sh.n_out * blocks * 136;
        std::vector<uint8_t> w(wbytes);
        for (auto& b : w) b = (uint8_t) rng();
        for (size_t i = 0; i < (size_t) sh.n_out * blocks; ++i) {   // d: a normal fp16 scale
            const uint16_t d = f16_bits(0.001f + 0.001f * (float) (rng() % 100));
            std::memcpy(&w[i * 136], &d, 2);
        }
        const int nq = sh.n_in / 32;
        std::vector<uint8_t> x((size_t) 8 * nq * 36);
        for (auto& b : x) b = (uint8_t) rng();
        for (int i = 0; i < 8 * nq; ++i) {
            const uint16_t d = f16_bits(0.01f), sum = f16_bits(0.5f);
            std::memcpy(&x[(size_t) i * 36], &d, 2);
            std::memcpy(&x[(size_t) i * 36 + 2], &sum, 2);
        }
        void *dw, *dx;
        float* dy;
        strata::gpu::alloc_device(&dw, wbytes);
        strata::gpu::alloc_device(&dx, x.size());
        strata::gpu::alloc_device(&dy, (size_t) 8 * sh.n_out * 4);
        strata::gpu::copy(dw, w.data(), wbytes);
        strata::gpu::copy(dx, x.data(), x.size());
        for (int nc = 1; nc <= 8; ++nc) {
            strata::kernels::native_iq4_xs_mmvq(dw, dx, dy, sh.n_in, sh.n_out, nc, s);
            strata::gpu::stream_sync(s);
            std::vector<float> y((size_t) nc * sh.n_out);
            strata::gpu::copy(y.data(), dy, y.size() * 4);
            all.insert(all.end(), y.begin(), y.end());
            const int reps = 200;
            const auto t0 = std::chrono::steady_clock::now();
            for (int r = 0; r < reps; ++r) strata::kernels::native_iq4_xs_mmvq(dw, dx, dy, sh.n_in, sh.n_out, nc, s);
            strata::gpu::stream_sync(s);
            const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / reps;
            std::printf("%-8s %5d -> %5d  cols %d  %8.1f us  %6.1f GB/s\n", sh.name, sh.n_in, sh.n_out, nc, us,
                        (double) wbytes / us / 1e3);
        }
        strata::gpu::free(dw);
        strata::gpu::free(dx);
        strata::gpu::free(dy);
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
        if (n != all.size()) { std::printf("check: %s has %zu outputs, want %zu\n", check.c_str(), n, all.size()); return 1; }
        size_t diff = 0;
        double worst = 0;
        for (size_t i = 0; i < n; ++i)
            if (std::memcmp(&ref[i], &all[i], 4) != 0) {
                ++diff;
                worst = std::max(worst, std::fabs((double) ref[i] - all[i]) / (std::fabs((double) ref[i]) + 1e-6));
            }
        std::printf("check: %zu of %zu outputs differ from %s, largest relative difference %.3g\n", diff, n,
                    check.c_str(), worst);
        return diff == 0 ? 0 : 2;
    }
    return 0;
}
