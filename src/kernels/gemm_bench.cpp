// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/gemm_bench.cpp - the prompt path's dense products' time on this GPU (XeStrata-890M): xmx_gemm (the
// matrix-engine kernel the GPU takes: XMX, mma_gemm's joint_matrix or WMMA, or DP4a) over Qwen3.8's projection
// shapes for a chunk of T rows, FP16 and BF16.  Random values from a fixed seed; `--check` compares every product
// against a host reference on a few rows (a layout change must keep the sums).
//     gemm_bench [--rows T] [--check] [--shape N,K]...   (default T 2048 and the model's shapes)
#include "strata/core/gpu.hpp"
#include "strata/kernels/xmx_gemm.hpp"

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
uint16_t f16_bits(float f) {   // normal range only: a value below FP16's smallest normal is made zero
    uint32_t u;
    std::memcpy(&u, &f, 4);
    const uint32_t s = (u >> 16) & 0x8000, m = (u >> 13) & 0x3ff;
    const int e = (int) ((u >> 23) & 0xff) - 127 + 15;
    if (e <= 0) return (uint16_t) s;
    return (uint16_t) (s | ((uint32_t) e << 10) | m);
}
uint16_t bf16_bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return (uint16_t) (u >> 16);
}
float f16_val(uint16_t h) {
    const uint32_t s = (h & 0x8000u) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    uint32_t u = e == 0 ? s : s | ((e + 127 - 15) << 23) | (m << 13);
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
float bf16_val(uint16_t h) {
    const uint32_t u = (uint32_t) h << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
template<class T> T* up(const std::vector<T>& v) {
    T* d = nullptr;
    strata::gpu::alloc_device(&d, v.size() * sizeof(T));
    strata::gpu::copy(d, v.data(), v.size() * sizeof(T));
    return d;
}
}  // namespace

int main(int argc, char** argv) {
    int64_t T = 2048;
    bool check = false;
    std::vector<std::pair<int64_t, int64_t>> shapes;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--rows") && i + 1 < argc) T = std::atoll(argv[++i]);
        else if (!std::strcmp(argv[i], "--check")) check = true;
        else if (!std::strcmp(argv[i], "--shape") && i + 1 < argc) {
            const std::string s = argv[++i];
            const auto c = s.find(',');
            if (c != std::string::npos) shapes.emplace_back(std::atoll(s.c_str()), std::atoll(s.c_str() + c + 1));
        }
    }
    if (shapes.empty())
        shapes = {{10240, 2560}, {6144, 2560}, {2560, 6144}, {12288, 2560}, {512, 2560}, {2560, 2560}, {640, 2560},
                  {2560, 640}, {320, 10240}, {10240, 320}, {128, 2560}, {48, 2560}};
    std::printf("gemm_bench: T %lld, xmx_gemm path FP16 %s, BF16 %s\n", (long long) T,
                strata::kernels::gemm_path(strata::kernels::XmxType::f16), strata::kernels::gemm_path(strata::kernels::XmxType::bf16));
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
    for (int bf = 0; bf < 2; ++bf) {
        const auto t = bf ? strata::kernels::XmxType::bf16 : strata::kernels::XmxType::f16;
        for (const auto& [N, K] : shapes) {
            std::vector<uint16_t> X((size_t) T * K), W((size_t) N * K);
            for (auto& v : X) v = bf ? bf16_bits(uni(rng)) : f16_bits(uni(rng));
            for (auto& v : W) v = bf ? bf16_bits(uni(rng) * 0.1f) : f16_bits(uni(rng) * 0.1f);
            uint16_t* dX = up(X);
            uint16_t* dW = up(W);
            float* dY = nullptr;
            strata::gpu::alloc_device(&dY, (size_t) T * N * 4);
            const bool ok = strata::kernels::xmx_gemm_ok(dX, dW, dY, N, K, N);
            auto run = [&] {
                if (ok) strata::kernels::xmx_gemm(t, dX, dW, dY, T, N, K, N, nullptr);
                else strata::kernels::gemm_rows(t, dX, dW, dY, T, N, K, N, nullptr);
                strata::gpu::stream_sync(nullptr);
            };
            run();
            run();
            const int reps = 5;
            const auto t0 = std::chrono::steady_clock::now();
            for (int r = 0; r < reps; ++r) run();
            const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / reps;
            const double tflops = 2.0 * (double) T * (double) N * (double) K / us / 1e6;
            std::string verdict;
            if (check) {
                std::vector<float> Y((size_t) T * N);
                strata::gpu::copy(Y.data(), dY, Y.size() * 4);
                double worst = 0;
                for (int64_t r : {(int64_t) 0, T / 2, T - 1}) {
                    for (int64_t n = 0; n < N; n += std::max<int64_t>(1, N / 64)) {
                        double ref = 0;
                        for (int64_t k = 0; k < K; ++k)
                            ref += (double) (bf ? bf16_val(X[(size_t) r * K + k]) : f16_val(X[(size_t) r * K + k])) *
                                   (double) (bf ? bf16_val(W[(size_t) n * K + k]) : f16_val(W[(size_t) n * K + k]));
                        worst = std::max(worst, std::fabs(ref - (double) Y[(size_t) r * N + n]) / (1.0 + std::fabs(ref)));
                    }
                }
                verdict = worst < 2e-3 ? " ok" : " WRONG";
                char b[32];
                std::snprintf(b, sizeof b, " (rel err %.1e)", worst);
                verdict += b;
            }
            std::printf("  %s N %5lld K %5lld: %9.1f us  %6.2f TFLOP/s%s%s\n", bf ? "BF16" : "FP16", (long long) N,
                        (long long) K, us, tflops, ok ? "" : " [gemm_rows]", verdict.c_str());
            strata::gpu::free(dX);
            strata::gpu::free(dW);
            strata::gpu::free(dY);
        }
    }
    return 0;
}
