// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/kernels/bf16_gemv_parity.cpp - P2.S5's test for the BF16 GEMV.
//
// THREE THINGS ARE CHECKED, and the third is the one the engine's correctness rests on:
//
//   1. both kernels against a host reference that reads the SAME bf16 bits - so what is under test is the
//      reduction, not the conversion, and the tolerance is set by summation order and nothing else;
//   2. the naive and the split kernel agree with each other, because the naive one is the reference the
//      split one exists to be checked against and a disagreement there means one of them is wrong;
//   3. **THE ACTIVATION CONTRACT IS OBSERVABLE.**  A BF16 weight sees a BF16 activation; feeding it fp16 or
//      f32 instead is a 2^-9 = 1.95e-03 relative change, which is above any 1e-3 tolerance.  An fp16
//      activation is the rival reading, computed and required to differ, because it is what the engine did
//      before `docs/activation-contract.md` settled the question and it produces perfectly plausible output.
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/f16_bits.hpp"

#include "parity_device.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

using strata::kernels::bf16_from_f32;
using strata::kernels::f32_from_bf16;

double rel_l1(const std::vector<float>& a, const std::vector<float>& b) {
    double d = 0, m = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        d += std::fabs((double) a[i] - (double) b[i]);
        m += std::fabs((double) a[i]);
    }
    return d / (m > 1e-30 ? m : 1e-30);
}

/// The host reference, in DOUBLE, over the same bf16 bits.
void reference(const std::vector<uint16_t>& x, const std::vector<uint16_t>& w, long long n_in, long long n_out,
               std::vector<float>& y) {
    y.assign((size_t) n_out, 0.0f);
    for (long long o = 0; o < n_out; ++o) {
        double acc = 0;
        for (long long i = 0; i < n_in; ++i)
            acc += (double) f32_from_bf16(x[(size_t) i]) * (double) f32_from_bf16(w[(size_t) (o * n_in + i)]);
        y[(size_t) o] = (float) acc;
    }
}

}  // namespace

int main(int argc, char** argv) {
    bool selftest = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--selftest") selftest = true;
        else { std::fprintf(stderr, "usage: bf16_gemv_parity [--selftest]\n"); return 2; }
    }
    int bad = 0;

    // THE REAL SHAPES, because a kernel that only works on a square toy is not a kernel this engine can use.
    // `ssm_alpha` is [2560, 48] - a wide reduction and 48 outputs - and `indexer.k_proj` is [2560, 128].
    struct Shape { long long n_in, n_out; const char* what; };
    const Shape shapes[] = {
        {2560, 48, "ssm_alpha.weight  [2560, 48]"},
        {2560, 128, "indexer.k_proj   [2560, 128]"},
        {2560, 512, "indexer.q_proj   [2560, 512]"},
        {2560, 2560, "ple_value        [2560, 2560]"},
    };

    for (const Shape& s : shapes) {
        std::mt19937 rng(4242);
        std::normal_distribution<float> g(0.0f, 1.0f);
        std::vector<float> fx((size_t) s.n_in), fw((size_t) (s.n_in * s.n_out));
        for (auto& v : fx) v = g(rng);
        // a realistic weight scale: these are projections, not embeddings
        for (auto& v : fw) v = g(rng) * 0.05f;

        // THE ACTIVATION IS ROUNDED TO BF16 FIRST, which is the contract, and the WEIGHT is already bf16 in
        // the pack - the loaded bytes ARE the high 16 bits of an f32.
        std::vector<uint16_t> x((size_t) s.n_in), w((size_t) (s.n_in * s.n_out));
        for (size_t i = 0; i < fx.size(); ++i) x[i] = bf16_from_f32(fx[i]);
        for (size_t i = 0; i < fw.size(); ++i) w[i] = bf16_from_f32(fw[i]);

        std::vector<float> want;
        reference(x, w, s.n_in, s.n_out, want);

        strata::parity::Device dev;
        uint16_t* d_x = dev.upload(x);
        uint16_t* d_w = dev.upload(w);
        float* d_y = dev.alloc<float>((size_t) s.n_out);

        std::vector<float> naive((size_t) s.n_out), warp((size_t) s.n_out), split((size_t) s.n_out);
        strata::kernels::bf16_gemv(d_x, d_w, d_y, s.n_in, s.n_out, nullptr);
        strata::parity::copy_out(naive.data(), d_y, naive.size());
        strata::kernels::bf16_gemv_split(d_x, d_w, d_y, s.n_in, s.n_out, 32, nullptr);
        strata::parity::copy_out(warp.data(), d_y, warp.size());
        strata::kernels::bf16_gemv_split(d_x, d_w, d_y, s.n_in, s.n_out, 256, nullptr);
        strata::parity::copy_out(split.data(), d_y, split.size());

        const double rn = rel_l1(want, naive), rw = rel_l1(want, warp), rs = rel_l1(want, split);
        std::printf("  %-34s naive %.2e  warp %.2e  split256 %.2e\n", s.what, rn, rw, rs);
        // The products are EXACT - both sides are bf16-valued, so f32 represents each product exactly - and
        // the only difference is the order the 2560 terms are added.  A tight bound is therefore meaningful.
        if (!(rn <= 1e-5 && rw <= 1e-5 && rs <= 1e-5)) { std::printf("    *** over 1e-5 ***\n"); ++bad; }

        // ---- 3. THE ACTIVATION CONTRACT.  The rival: an fp16 activation instead of bf16, which is what the
        // engine did before the contract was settled.  It must be observable, or check 1 above would pass
        // against either and the contract would be untested.
        {
            std::vector<float> fp16_as_f32((size_t) s.n_in);
            std::vector<uint16_t> x16((size_t) s.n_in);
            for (size_t i = 0; i < fx.size(); ++i) {
                x16[i] = strata::kernels::f16_from_f32(fx[i]);
                fp16_as_f32[i] = strata::kernels::f32_from_f16(x16[i]);
            }
            // Round the fp16 value to bf16 - i.e. give the kernel a bf16 activation whose VALUES came through
            // fp16.  That is the whole difference between the two contracts at this call site.
            for (size_t i = 0; i < fp16_as_f32.size(); ++i) x[i] = bf16_from_f32(fp16_as_f32[i]);
            strata::parity::copy_in(d_x, x.data(), x.size());
            std::vector<float> rival((size_t) s.n_out);
            strata::kernels::bf16_gemv(d_x, d_w, d_y, s.n_in, s.n_out, nullptr);
            strata::parity::copy_out(rival.data(), d_y, rival.size());
            const double r = rel_l1(want, rival);
            const bool visible = r > 1e-4;
            std::printf("      %-30s %s (%.4f%% apart)\n", "bf16 vs fp16 activation",
                        visible ? "yes" : "*** NO ***", r * 100);
            if (!visible) ++bad;
        }
    }

    // ---- 4. THE WINDOW'S ROWS IN ONE LAUNCH (bf16_gemv_fp32_mmvf_multi, the decode window's form): F32 activations,
    // 2..8 rows.  Against the host's FP64 reference over the same values, against the single-row call (the header says
    // every output is that call's bits: memcmp), and the run-time skip: while *skip_a <= *skip_b the launch writes
    // nothing, and otherwise what the launch without the pointers writes.
    for (const Shape& s : {shapes[1], shapes[2]}) {
        std::mt19937 rng(2051);  // NOLINT(bugprone-random-generator-seed): reproducible fixtures.
        std::normal_distribution<float> g(0.0f, 1.0f);
        const int max_rows = 8;
        std::vector<float> fx((size_t) (max_rows * s.n_in));
        std::vector<uint16_t> w((size_t) (s.n_in * s.n_out));
        for (auto& v : fx) v = g(rng);
        for (auto& v : w) v = bf16_from_f32(g(rng) * 0.05f);
        strata::parity::Device dev;
        float* d_x = dev.upload(fx);
        uint16_t* d_w = dev.upload(w);
        float* d_y = dev.alloc<float>((size_t) (max_rows * s.n_out));
        float* d_one = dev.alloc<float>((size_t) s.n_out);
        int32_t* d_flag = dev.alloc<int32_t>(2);
        float* d_aux_y = dev.alloc<float>((size_t) max_rows);
        int shape_bad = 0, aux_ran = 0;
        double worst = 0.0;
        for (int n = 2; n <= max_rows; ++n) {
            const size_t count = (size_t) (n * s.n_out);
            std::vector<float> got(count), single(count), want(count);
            strata::kernels::bf16_gemv_fp32_mmvf_multi(d_x, s.n_in, d_w, d_y, s.n_out, s.n_in, s.n_out, n, nullptr);
            strata::parity::copy_out(got.data(), d_y, count);
            for (int t = 0; t < n; ++t) {
                strata::kernels::bf16_gemv_fp32_mmvf(d_x + (size_t) (t * s.n_in), d_w, d_one, (int) s.n_in, (int) s.n_out, nullptr);
                strata::parity::copy_out(single.data() + (size_t) (t * s.n_out), d_one, (size_t) s.n_out);
                for (long long o = 0; o < s.n_out; ++o) {
                    double sum = 0.0;
                    for (long long i = 0; i < s.n_in; ++i)
                        sum += (double) strata::kernels::f32_from_bf16(w[(size_t) (o * s.n_in + i)]) *
                               (double) fx[(size_t) (t * s.n_in + i)];
                    want[(size_t) (t * s.n_out + o)] = (float) sum;
                }
            }
            worst = std::max(worst, rel_l1(want, got));
            if (std::memcmp(got.data(), single.data(), count * sizeof(float)) != 0) {
                std::printf("    *** %d rows: not the single-row call's bits ***\n", n);
                ++shape_bad;
            }
            // the skip: taken (5 <= 5), then not taken (6 <= 5 is false)
            const std::vector<float> mark(count, -12345.0f);
            const int32_t taken[2] = {5, 5}, not_taken[2] = {6, 5};
            std::vector<float> kept(count), ran(count);
            strata::parity::copy_in(d_y, mark.data(), count);
            strata::parity::copy_in(d_flag, taken, 2);
            strata::kernels::bf16_gemv_fp32_mmvf_multi(d_x, s.n_in, d_w, d_y, s.n_out, s.n_in, s.n_out, n, nullptr, d_flag,
                                                       d_flag + 1);
            strata::parity::copy_out(kept.data(), d_y, count);
            strata::parity::copy_in(d_flag, not_taken, 2);
            strata::kernels::bf16_gemv_fp32_mmvf_multi(d_x, s.n_in, d_w, d_y, s.n_out, s.n_in, s.n_out, n, nullptr, d_flag,
                                                       d_flag + 1);
            strata::parity::copy_out(ran.data(), d_y, count);
            if (std::memcmp(kept.data(), mark.data(), count * sizeof(float)) != 0) {
                std::printf("    *** %d rows: the skipped launch wrote ***\n", n);
                ++shape_bad;
            }
            if (std::memcmp(ran.data(), got.data(), count * sizeof(float)) != 0) {
                std::printf("    *** %d rows: the launch with the skip pointers differs ***\n", n);
                ++shape_bad;
            }
            // the product and one more row in the same launch (bf16_gemv_fp32_mmvf_multi_aux: the router and the
            // shared expert's scalar gate), here the matrix's last row again: the product's bits, and for the extra
            // row the bits the product has for that row.  It launches nothing where the product takes the 4-rows form.
            strata::parity::copy_in(d_y, mark.data(), count);
            if (strata::kernels::bf16_gemv_fp32_mmvf_multi_aux(d_x, s.n_in, d_w, d_y, s.n_out, s.n_in, s.n_out, n,
                                                               d_w + (size_t) ((s.n_out - 1) * s.n_in), d_aux_y, 1,
                                                               nullptr)) {
                std::vector<float> with_aux(count), aux_got((size_t) n), aux_want((size_t) n);
                strata::parity::copy_out(with_aux.data(), d_y, count);
                strata::parity::copy_out(aux_got.data(), d_aux_y, (size_t) n);
                for (int t = 0; t < n; ++t) aux_want[(size_t) t] = got[(size_t) (t * s.n_out + s.n_out - 1)];
                if (std::memcmp(with_aux.data(), got.data(), count * sizeof(float)) != 0 ||
                    std::memcmp(aux_got.data(), aux_want.data(), (size_t) n * sizeof(float)) != 0) {
                    std::printf("    *** %d rows: the launch with the extra row differs ***\n", n);
                    ++shape_bad;
                }
                ++aux_ran;
            }
        }
        std::printf("  %-34s rows 2..8 %.2e, the single row's bits, the skip and the extra row (%d widths) %s\n",
                    s.what, worst, aux_ran, shape_bad ? "*** NO ***" : "yes");
        if (!(worst <= 1e-5)) { std::printf("    *** over 1e-5 ***\n"); ++shape_bad; }
        bad += shape_bad;
    }

    std::printf("\nbf16_gemv: %d failures\n", bad);
    if (bad) return 1;
    if (selftest) std::printf("bf16_gemv_parity OK\n");
    return 0;
}
