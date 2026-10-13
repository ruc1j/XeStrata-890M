// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/gdn_bench.cpp - the GDN state's per-window time on this GPU (XeStrata-890M): the verify window's step
// (outputs, the state read) and the commit (the state updated by the kept rows) at the model's shape (S 128, 16 key
// heads, 48 value heads), over 8 layers' states round robin so the L2 does not hold them.  Fixed-seed values:
// `--save F` writes every layer's state after the runs, `--check F` compares a changed kernel's to them.
//     gdn_bench [--save F | --check F]
#include "strata/core/gpu.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    std::string save, check;
    for (int i = 1; i + 1 < argc; ++i) {
        if (!std::strcmp(argv[i], "--save")) save = argv[++i];
        else if (!std::strcmp(argv[i], "--check")) check = argv[++i];
    }
    constexpr int S = 128, HK = 16, HV = 48, C = 2 * S * HK + S * HV, MT = 8, LAYERS = 8;
    const size_t n_state = (size_t) S * S * HV;
    std::mt19937 rng(5);
    std::uniform_real_distribution<float> u(-0.05f, 0.05f);
    auto fill = [&](size_t n) { std::vector<float> v(n); for (auto& x : v) x = u(rng); return v; };
    const auto st = fill(n_state * LAYERS), h = fill((size_t) MT * C), z = fill((size_t) MT * S * HV), gm = fill(S);
    std::vector<float> gate(MT * HV), beta(MT * HV);
    for (auto& x : gate) x = -0.1f + u(rng);
    for (auto& x : beta) x = 0.5f + u(rng);
    float *d_st, *d_h, *d_g, *d_b, *d_z, *d_gm, *d_y;
    int32_t* d_keep;
    strata::gpu::alloc_device(&d_st, st.size() * 4);
    strata::gpu::alloc_device(&d_h, h.size() * 4);
    strata::gpu::alloc_device(&d_g, gate.size() * 4);
    strata::gpu::alloc_device(&d_b, beta.size() * 4);
    strata::gpu::alloc_device(&d_z, z.size() * 4);
    strata::gpu::alloc_device(&d_gm, gm.size() * 4);
    strata::gpu::alloc_device(&d_y, z.size() * 4);
    strata::gpu::alloc_device(&d_keep, 4);
    strata::gpu::copy(d_st, st.data(), st.size() * 4);
    strata::gpu::copy(d_h, h.data(), h.size() * 4);
    strata::gpu::copy(d_g, gate.data(), gate.size() * 4);
    strata::gpu::copy(d_b, beta.data(), beta.size() * 4);
    strata::gpu::copy(d_z, z.data(), z.size() * 4);
    strata::gpu::copy(d_gm, gm.data(), gm.size() * 4);
    void* s = strata::gpu::stream_create();
    strata::kernels::gdn_step_tune(C, HK, HV, s);
    for (int T = 1; T <= 6; ++T) {
        const int32_t keep = T;   // every row kept
        strata::gpu::copy(d_keep, &keep, 4);
        const int reps = 64;
        double us_step = 0, us_commit = 0;
        for (int pass = 0; pass < 2; ++pass) {
            const auto t0 = std::chrono::steady_clock::now();
            for (int r = 0; r < reps; ++r) {
                float* stl = d_st + (size_t) (r % LAYERS) * n_state;
                if (pass == 0)
                    strata::kernels::gdn_step_norm_multi(stl, d_h, C, d_g, d_b, d_z, d_gm, 1e-6f, d_y, HK, HV, T, nullptr, s);
                else
                    strata::kernels::gdn_step_norm_multi(stl, d_h, C, d_g, d_b, d_z, d_gm, 1e-6f, d_y, HK, HV, T, d_keep, s, T);
            }
            strata::gpu::stream_sync(s);
            const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / reps;
            (pass == 0 ? us_step : us_commit) = us;
        }
        std::printf("T %d  step %7.1f us  commit %7.1f us  (state %.1f MB a layer)\n", T, us_step, us_commit,
                    n_state * 4 / 1e6);
    }
    std::vector<float> out(st.size());
    strata::gpu::copy(out.data(), d_st, out.size() * 4);
    if (!save.empty()) {
        FILE* f = std::fopen(save.c_str(), "wb");
        std::fwrite(out.data(), 4, out.size(), f);
        std::fclose(f);
        std::printf("saved %zu values\n", out.size());
    }
    if (!check.empty()) {
        std::vector<float> ref(out.size());
        FILE* f = std::fopen(check.c_str(), "rb");
        const size_t n = f ? std::fread(ref.data(), 4, ref.size(), f) : 0;
        if (f) std::fclose(f);
        size_t diff = 0;
        for (size_t i = 0; i < n; ++i) diff += std::memcmp(&ref[i], &out[i], 4) != 0;
        std::printf("check: %zu of %zu values differ\n", diff + (out.size() - n), out.size());
        return diff == 0 && n == out.size() ? 0 : 2;
    }
    return 0;
}
