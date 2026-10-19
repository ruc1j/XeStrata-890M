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
        double us_step = 0, us_commit = 0, us_both = 0;
        for (int pass = 0; pass < 3; ++pass) {
            const auto t0 = std::chrono::steady_clock::now();
            for (int r = 0; r < reps; ++r) {
                float* stl = d_st + (size_t) (r % LAYERS) * n_state;
                if (pass == 0)
                    strata::kernels::gdn_step_norm_multi(stl, d_h, C, d_g, d_b, d_z, d_gm, 1e-6f, d_y, HK, HV, T, nullptr, s);
                else if (pass == 2)   // the outputs and the state written by the same launch
                    strata::kernels::gdn_step_norm_multi(stl, d_h, C, d_g, d_b, d_z, d_gm, 1e-6f, d_y, HK, HV, T, d_keep, s);
                else
                    strata::kernels::gdn_step_norm_multi(stl, d_h, C, d_g, d_b, d_z, d_gm, 1e-6f, d_y, HK, HV, T, d_keep, s, T);
            }
            strata::gpu::stream_sync(s);
            const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / reps;
            (pass == 0 ? us_step : pass == 1 ? us_commit : us_both) = us;
        }
        std::printf("T %d  step %7.1f us  commit %7.1f us  step+write %7.1f us  (state %.1f MB a layer)\n", T, us_step,
                    us_commit, us_both, n_state * 4 / 1e6);
    }
    {   // the commit folded into the next window (gdn_step_norm_pending) against step + commit + step: the second
        // window's outputs and the state after it, bit for bit, and the time a layer of each
        const int TA = 4, NA = 3, TB = 3;
        float *d_s1, *d_s2, *d_y1, *d_y2, *d_ph, *d_pg, *d_pb;
        int32_t *d_na, *d_pn;
        strata::gpu::alloc_device(&d_s1, n_state * 4);
        strata::gpu::alloc_device(&d_s2, n_state * 4);
        strata::gpu::alloc_device(&d_y1, z.size() * 4);
        strata::gpu::alloc_device(&d_y2, z.size() * 4);
        strata::gpu::alloc_device(&d_ph, h.size() * 4);
        strata::gpu::alloc_device(&d_pg, gate.size() * 4);
        strata::gpu::alloc_device(&d_pb, beta.size() * 4);
        strata::gpu::alloc_device(&d_na, 4);
        strata::gpu::alloc_device(&d_pn, 4);
        const int32_t na = NA, zero = 0;
        strata::gpu::copy(d_na, &na, 4);
        strata::gpu::copy(d_pn, &zero, 4);
        strata::gpu::copy(d_s1, st.data(), n_state * 4);
        strata::gpu::copy(d_s2, st.data(), n_state * 4);
        const strata::kernels::GdnPending pd{d_ph, d_pg, d_pb, d_pn};
        // A then commit, B: the old way
        strata::kernels::gdn_step_norm_multi(d_s1, d_h, C, d_g, d_b, d_z, d_gm, 1e-6f, d_y1, HK, HV, TA, nullptr, s);
        strata::kernels::gdn_step_norm_multi(d_s1, d_h, C, d_g, d_b, d_z, d_gm, 1e-6f, d_y1, HK, HV, TA, d_na, s, TA);
        strata::kernels::gdn_step_norm_multi(d_s1, d_h + C, C, d_g + HV, d_b + HV, d_z, d_gm, 1e-6f, d_y1, HK, HV, TB, nullptr, s);
        // A then stash, B with the pending rows
        strata::kernels::gdn_step_norm_pending(d_s2, d_h, C, d_g, d_b, d_z, d_gm, 1e-6f, d_y2, HK, HV, TA, nullptr, s, 0,
                                               nullptr, pd);
        strata::kernels::gdn_pending_stash(d_h, d_g, d_b, C, HK, HV, MT, 1, d_na, d_ph, d_pg, d_pb, d_pn, s);
        strata::kernels::gdn_step_norm_pending(d_s2, d_h + C, C, d_g + HV, d_b + HV, d_z, d_gm, 1e-6f, d_y2, HK, HV, TB,
                                               nullptr, s, 0, nullptr, pd);
        strata::gpu::stream_sync(s);
        std::vector<float> a(n_state), b(n_state), ya((size_t) TB * S * HV), yb(ya.size());
        strata::gpu::copy(a.data(), d_s1, n_state * 4);
        strata::gpu::copy(b.data(), d_s2, n_state * 4);
        strata::gpu::copy(ya.data(), d_y1, ya.size() * 4);
        strata::gpu::copy(yb.data(), d_y2, yb.size() * 4);
        size_t ds = 0, dy = 0;
        for (size_t i = 0; i < n_state; ++i) ds += std::memcmp(&a[i], &b[i], 4) != 0;
        for (size_t i = 0; i < ya.size(); ++i) dy += std::memcmp(&ya[i], &yb[i], 4) != 0;
        // and a flush of a stash: the commit's state
        strata::kernels::gdn_pending_stash(d_h + C, d_g + HV, d_b + HV, C, HK, HV, MT, 1, d_na, d_ph, d_pg, d_pb, d_pn, s);
        strata::kernels::gdn_pending_flush(d_s2, 0, C, HK, HV, MT, 1, pd, s);
        strata::kernels::gdn_step_norm_multi(d_s1, d_h + C, C, d_g + HV, d_b + HV, d_z, d_gm, 1e-6f, d_y1, HK, HV, MT, d_na, s, MT);
        strata::gpu::stream_sync(s);
        strata::gpu::copy(a.data(), d_s1, n_state * 4);
        strata::gpu::copy(b.data(), d_s2, n_state * 4);
        size_t df = 0;
        for (size_t i = 0; i < n_state; ++i) df += std::memcmp(&a[i], &b[i], 4) != 0;
        std::printf("pending: B's outputs %zu, state %zu, flushed state %zu values differ\n", dy, ds, df);
        // time a layer: step + commit against the step with 3 pending rows (the stash of 36 layers / 36 included)
        strata::gpu::copy(d_pn, &na, 4);   // 3 rows pending in every step below
        for (int T = 1; T <= 4; ++T) {
            const int reps = 64;
            double us[2] = {};
            for (int pass = 0; pass < 2; ++pass) {
                const auto t0 = std::chrono::steady_clock::now();
                for (int r = 0; r < reps; ++r) {
                    float* stl = d_st + (size_t) (r % LAYERS) * n_state;
                    if (pass == 0) {
                        strata::kernels::gdn_step_norm_multi(stl, d_h, C, d_g, d_b, d_z, d_gm, 1e-6f, d_y, HK, HV, T, nullptr, s);
                        strata::kernels::gdn_step_norm_multi(stl, d_h, C, d_g, d_b, d_z, d_gm, 1e-6f, d_y, HK, HV, T, d_na, s, T);
                    } else {
                        strata::kernels::gdn_step_norm_pending(stl, d_h, C, d_g, d_b, d_z, d_gm, 1e-6f, d_y, HK, HV, T,
                                                               nullptr, s, 0, nullptr, pd);
                    }
                }
                strata::gpu::stream_sync(s);
                us[pass] = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / reps;
            }
            std::printf("T %d  step + commit %7.1f us  step with 3 pending rows %7.1f us\n", T, us[0], us[1]);
        }
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
