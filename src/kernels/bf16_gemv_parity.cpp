// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/kernels/bf16_gemv_parity.cpp - P2.S5's test for the BF16 GEMV (bf16_gemv_fp32_mmvf: BF16 weights, FP32
// activations, the pinned CUDA single-token MMVF).
//
// THREE THINGS ARE CHECKED, and the second is the one the engine's correctness rests on:
//
//   1. the kernel against a host reference in double over the SAME bf16 weight bits and FP32 activations - so the
//      tolerance is set by f32 products and summation order and nothing else;
//   2. **THE ACTIVATION CONTRACT IS OBSERVABLE.**  The kernel reads the activation in FP32; a BF16-rounded
//      activation is a 2^-9 = 1.95e-03 relative change, the rival reading (the engine's former contract), computed
//      and required to differ, because it produces perfectly plausible output.
//   3. the F32 MMVF against the image engine's former f32_vec (mf_dit.cpp, a work-group of 32 striding a row),
//      now taken to the MMVF and a bias add: the sums' order differs, so the difference's size is the check.
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/window_limits.hpp"

#include "parity_device.hpp"

#include <sycl/sycl.hpp>

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

/// The host reference, in DOUBLE, over the same bf16 weight bits and the given activations.
void reference(const std::vector<float>& x, const std::vector<uint16_t>& w, long long n_in, long long n_out,
               std::vector<float>& y) {
    y.assign((size_t) n_out, 0.0f);
    for (long long o = 0; o < n_out; ++o) {
        double acc = 0;
        for (long long i = 0; i < n_in; ++i)
            acc += (double) x[(size_t) i] * (double) f32_from_bf16(w[(size_t) (o * n_in + i)]);
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
        std::mt19937 rng(4242);  // NOLINT(bugprone-random-generator-seed): a fixed seed keeps the parity inputs reproducible
        std::normal_distribution<float> g(0.0f, 1.0f);
        std::vector<float> fx((size_t) s.n_in), fw((size_t) (s.n_in * s.n_out));
        for (auto& v : fx) v = g(rng);
        // a realistic weight scale: these are projections, not embeddings
        for (auto& v : fw) v = g(rng) * 0.05f;

        // THE WEIGHT is already bf16 in the pack - the loaded bytes ARE the high 16 bits of an f32; the activation
        // stays f32.
        std::vector<uint16_t> w((size_t) (s.n_in * s.n_out));
        for (size_t i = 0; i < fw.size(); ++i) w[i] = bf16_from_f32(fw[i]);

        std::vector<float> want;
        reference(fx, w, s.n_in, s.n_out, want);

        strata::parity::Device dev;
        float* d_x = dev.upload(fx);
        uint16_t* d_w = dev.upload(w);
        float* d_y = dev.alloc<float>((size_t) s.n_out);

        std::vector<float> got((size_t) s.n_out);
        strata::kernels::bf16_gemv_fp32_mmvf(d_x, d_w, d_y, s.n_in, s.n_out, nullptr);
        strata::parity::copy_out(got.data(), d_y, got.size());

        const double r = rel_l1(want, got);
        std::printf("  %-34s mmvf %.2e\n", s.what, r);
        // f32 products of an f32 activation and a bf16 weight, summed in f32 over n_in terms: a tight bound
        if (!(r <= 1e-5)) { std::printf("    *** over 1e-5 ***\n"); ++bad; }

        // ---- 2. THE ACTIVATION CONTRACT.  The rival: the activation rounded to bf16 first.  It must be observable,
        // or check 1 above would pass against either and the contract would be untested.
        {
            std::vector<float> xb((size_t) s.n_in);
            for (size_t i = 0; i < fx.size(); ++i) xb[i] = f32_from_bf16(bf16_from_f32(fx[i]));
            std::vector<float> rival;
            reference(xb, w, s.n_in, s.n_out, rival);
            const double rv = rel_l1(rival, got);
            const bool visible = rv > 1e-4;
            std::printf("      %-30s %s (%.4f%% apart)\n", "fp32 vs bf16 activation",
                        visible ? "yes" : "*** NO ***", rv * 100);
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

    // ---- the multi-row calls over a wide window: 16 rows in one call (the 16-row instances) against two calls of 8
    // (the instances a window of up to 8 rows takes), bit for bit, BF16 and F32 weights
    {
        const long long n_in = 2560, n_out = 512;
        const int T = strata::kernels::kWindowMaxT, H = strata::kernels::kVerifyNarrowT;
        std::mt19937 rng(1616);  // NOLINT(bugprone-random-generator-seed): a fixed seed keeps the parity inputs reproducible
        std::normal_distribution<float> g(0.0f, 1.0f);
        std::vector<float> fx((size_t) (T * n_in)), fw((size_t) (n_in * n_out));
        for (auto& v : fx) v = g(rng);
        for (auto& v : fw) v = g(rng) * 0.05f;
        std::vector<uint16_t> w((size_t) (n_in * n_out));
        for (size_t i = 0; i < fw.size(); ++i) w[i] = bf16_from_f32(fw[i]);
        strata::parity::Device dev;
        float *d_x = dev.upload(fx), *d_fw = dev.upload(fw);
        uint16_t* d_w = dev.upload(w);
        float *d_a = dev.alloc<float>((size_t) (T * n_out)), *d_b = dev.alloc<float>((size_t) (T * n_out));
        std::vector<float> a((size_t) (T * n_out)), b((size_t) (T * n_out));
        for (int f32 = 0; f32 < 2; ++f32) {
            auto run = [&](float* y, int t0, int n) {
                if (f32) strata::kernels::f32_gemv_fp32_mmvf_multi(d_x + t0 * n_in, n_in, d_fw, y + t0 * n_out, n_out,
                                                                   n_in, n_out, n, nullptr);
                else strata::kernels::bf16_gemv_fp32_mmvf_multi(d_x + t0 * n_in, n_in, d_w, y + t0 * n_out, n_out, n_in,
                                                                n_out, n, nullptr);
            };
            run(d_a, 0, T);
            run(d_b, 0, H);
            run(d_b, H, T - H);
            strata::parity::copy_out(a.data(), d_a, a.size());
            strata::parity::copy_out(b.data(), d_b, b.size());
            const bool same = a == b;
            std::printf("  %-34s %s\n", f32 ? "F32 multi, 16 rows vs 8 + 8" : "BF16 multi, 16 rows vs 8 + 8",
                        same ? "bitwise" : "*** DIFFER ***");
            if (!same) ++bad;
        }
    }

    // ---- 3. THE IMAGE ENGINE'S FORMER f32_vec (mf_dit.cpp; now the F32 MMVF and a bias add).
    // The MMVF sums in its own tuned order, so the bytes may differ in the last places: the check is the
    // difference's size over MageTrail's low-rank modulation shapes (the only reader f32_vec had).
    {
        auto old_f32_vec = [](const float* w, const float* x, const float* b, float* y, int64_t N, int64_t K) {
            auto& q = strata::core::Runtime::get().compute();
            constexpr int64_t local = 32;
            q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (N * local)), sycl::range<1>((size_t) local)),
                           [=](sycl::nd_item<1> it) {
                const auto n = static_cast<int64_t>(it.get_group_linear_id());
                const auto lane = static_cast<int64_t>(it.get_local_linear_id());
                const float* row = w + n * K;
                float acc = 0.f;
                for (int64_t k = lane; k < K; k += local) acc += row[k] * x[k];
                acc = sycl::reduce_over_group(it.get_group(), acc, sycl::plus<float>());
                if (lane == 0) y[n] = acc + (b != nullptr ? b[n] : 0.f);
            });
            q.wait();
        };
        struct FShape { int64_t K, N; const char* what; };
        const FShape fshapes[] = {
            {3072, 256, "mdown  [3072 -> 256]"},
            {256, 18432, "mod    [256 -> 6 x 3072]"},
            {512, 3072, "mod    [512 -> 3072]"},
        };
        for (const FShape& s : fshapes) {
            std::mt19937 rng(4243);  // NOLINT(bugprone-random-generator-seed): a fixed seed keeps the inputs reproducible
            std::normal_distribution<float> g(0.0f, 1.0f);
            std::vector<float> fx((size_t) s.K), fw((size_t) (s.K * s.N)), fb((size_t) s.N);
            for (auto& v : fx) v = g(rng);
            for (auto& v : fw) v = g(rng) * 0.05f;
            for (auto& v : fb) v = g(rng) * 0.1f;
            strata::parity::Device dev;
            float* d_x = dev.upload(fx);
            float* d_w = dev.upload(fw);
            float* d_b = dev.upload(fb);
            float* d_old = dev.alloc<float>((size_t) s.N);
            float* d_new = dev.alloc<float>((size_t) s.N);
            old_f32_vec(d_w, d_x, d_b, d_old, s.N, s.K);
            strata::kernels::f32_gemv_fp32_mmvf(d_x, d_w, d_new, s.K, s.N, nullptr);
            strata::kernels::add_inplace(d_new, d_b, s.N, nullptr, 1.0f);
            std::vector<float> o((size_t) s.N), nw((size_t) s.N);
            strata::parity::copy_out(o.data(), d_old, o.size());
            strata::parity::copy_out(nw.data(), d_new, nw.size());
            double mx = 0, den = 0;
            for (size_t i = 0; i < o.size(); ++i) {
                mx = std::max(mx, std::fabs((double) o[i] - nw[i]));
                den = std::max(den, std::fabs((double) o[i]));
            }
            const double r = rel_l1(o, nw);
            std::printf("  %-34s old vs mmvf+bias: rel %.2e, max %.2e (of %.2e)\n", s.what, r, mx, den);
            if (!(r <= 1e-5)) { std::printf("    *** over 1e-5 ***\n"); ++bad; }
        }
    }

    std::printf("\nbf16_gemv: %d failures\n", bad);
    if (bad) return 1;
    if (selftest) std::printf("bf16_gemv_parity OK\n");
    return 0;
}
