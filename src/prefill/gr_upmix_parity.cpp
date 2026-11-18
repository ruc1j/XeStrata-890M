// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/prefill/gr_upmix_parity.cpp - the hyper-connection read's up projection and mix in one kernel (gr_upmix,
// STRATA_HC_UPMIX=1 on gfx1150, after nicejunjie/Strata-rdna3's S23 and its gr_upmix_parity) against two references
// (GPU, synthetic, no model):
//
//   1. a float64 oracle: gated = (lo16 + lo16_lo) . w_up in float64, then the mix in float64.  Every output must lie
//      within the bound derived beforehand, per (t, d):
//          B = (1/HC) * sum_c |x_c| * (1/4 * 2K * 2^-24 * S_c + 4 * 2^-24) + 4 * 2^-24 * |mixed|,
//      x_c = R * rs * w (the normalized value), S_c = the sum of |a b| over the 2K products of stream c's dot (the
//      FP32 sums' error, through sigmoid's slope of at most 1/4), and a few FP32 roundings of the epilogue;
//   2. gr_mix_r on the GPU with `gated` from the oracle rounded to FP32: reported (the two differ by the GEMM's
//      summation order only);
// then the images: mixed16 is bf16(mixed), mixed16_lo bf16(mixed - mixed16) and mixed_h fp16(mixed), bit for bit; the
// outputs past T keep a canary; and a negative control - the same bound check over an output moved by 2.5B - must
// fail.  Cases: T = 300 (not a multiple of the 256 tokens a work-group takes, nor of 16), T = 37, T = 1, with and
// without the low parts.
// Usage: gr_upmix_parity   (exit 0: all pass; 77: gr_upmix does not run on this device)
#include "strata/core/geometry.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/prefill/kernels.hpp"

#include "../kernels/parity_device.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace par = strata::parity;
namespace pf = strata::prefill;
using strata::kernels::bf16_from_f32;
using strata::kernels::f32_from_bf16;

namespace {

constexpr int64_t N = strata::core::kCompiledGeometry.n_embd, HC = strata::core::kCompiledGeometry.hc, D = N * HC,
                  LR = strata::core::kCompiledGeometry.hc_lr;

int g_fail = 0;
void check(bool ok, const char* what) {
    if (!ok) ++g_fail;
    std::printf("  %-62s %s\n", what, ok ? "ok" : "FAIL");
}

double sigm(double x) { return 1.0 / (1.0 + std::exp(-x)); }

}  // namespace

int main() {
    setenv("STRATA_HC_UPMIX", "1", 1);
    par::Device dev;
    if (!pf::gr_upmix_on(par::stream())) {
        std::printf("SKIP: gr_upmix does not run on this device (gfx1150 only)\n");
        return 77;
    }
    std::mt19937 rng(4242);   // NOLINT(bugprone-random-generator-seed): the same data each run.
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_real_distribution<float> ur(0.5f, 1.5f);
    // w_up [D, LR] in BF16, shared by the cases
    std::vector<uint16_t> wu((size_t) (D * LR));
    for (auto& x : wu) x = bf16_from_f32(nd(rng) * 0.06f);
    const uint16_t* d_wu = dev.upload(wu);
    std::vector<float> wn((size_t) D);
    for (auto& x : wn) x = ur(rng);
    const float* d_wn = dev.upload(wn);

    struct Case { const char* name; int64_t T; bool lo; };
    const Case cases[] = {
        {"T = 300 (ragged), with the low parts", 300, true},
        {"T = 300 (ragged), high parts only", 300, false},
        {"T = 37", 37, true},
        {"T = 1", 1, true},
    };
    const float CANARY = -7777.25f;
    for (const Case& cs : cases) {
        const int64_t T = cs.T;
        std::printf("case: %s\n", cs.name);
        // lo16 as silu's outputs are: small positive and negative values; lo16_lo the low parts of FP32 values
        std::vector<uint16_t> lo16((size_t) (T * LR)), lo16_lo((size_t) (T * LR));
        for (size_t i = 0; i < lo16.size(); ++i) {
            const float v = nd(rng) * 0.7f;
            lo16[i] = bf16_from_f32(v);
            lo16_lo[i] = bf16_from_f32(v - f32_from_bf16(lo16[i]));
        }
        std::vector<float> R((size_t) (T * D)), rs((size_t) (T * HC));
        for (auto& x : R) x = nd(rng) * 3.0f;
        for (auto& x : rs) x = ur(rng) * 0.3f;
        // the oracle: gated in float64, then the mix in float64
        std::vector<double> gated((size_t) (T * D)), S((size_t) (T * D));
        for (int64_t t = 0; t < T; ++t)
            for (int64_t j = 0; j < D; ++j) {
                double a = 0, s = 0;
                const uint16_t* wr = &wu[(size_t) (j * LR)];
                for (int64_t k = 0; k < LR; ++k) {
                    const double w = f32_from_bf16(wr[k]);
                    const double hi = f32_from_bf16(lo16[(size_t) (t * LR + k)]) * w;
                    a += hi;
                    s += std::fabs(hi);
                    if (cs.lo) {
                        const double lo = f32_from_bf16(lo16_lo[(size_t) (t * LR + k)]) * w;
                        a += lo;
                        s += std::fabs(lo);
                    }
                }
                gated[(size_t) (t * D + j)] = a;
                S[(size_t) (t * D + j)] = s;
            }
        std::vector<double> mix_ref((size_t) (T * N)), bound((size_t) (T * N));
        const double u = std::ldexp(1.0, -24), K2 = (double) (cs.lo ? 2 * LR : LR);
        for (int64_t t = 0; t < T; ++t)
            for (int64_t d = 0; d < N; ++d) {
                double m = 0, b = 0;
                for (int64_t c = 0; c < HC; ++c) {
                    const size_t j = (size_t) (t * D + c * N + d);
                    const double x = (double) R[j] * rs[(size_t) (t * HC + c)] * wn[(size_t) (c * N + d)];
                    m += x * sigm(gated[j]);
                    b += std::fabs(x) * (0.25 * K2 * u * S[j] + 4 * u);
                }
                m /= (double) HC;
                mix_ref[(size_t) (t * N + d)] = m;
                bound[(size_t) (t * N + d)] = b / (double) HC + 4 * u * std::fabs(m);
            }

        const uint16_t* d_lo16 = dev.upload(lo16);
        const uint16_t* d_lo16_lo = cs.lo ? dev.upload(lo16_lo) : nullptr;
        const float* d_R = dev.upload(R);
        const float* d_rs = dev.upload(rs);
        const int64_t PAD = 4096;   // outputs past T: a canary the kernel must not touch
        std::vector<float> mixed((size_t) (T * N + PAD), CANARY);
        std::vector<uint16_t> m16((size_t) (T * N + PAD), 0xABCD), m16lo(m16), mh(m16);
        float* d_mixed = dev.upload(mixed);
        uint16_t* d_m16 = dev.upload(m16);
        uint16_t* d_m16lo = dev.upload(m16lo);
        uint16_t* d_mh = dev.upload(mh);
        pf::gr_upmix(d_lo16, d_lo16_lo, d_wu, d_R, d_rs, d_wn, d_mixed, d_m16, T, par::stream(), d_mh, d_m16lo);
        // the default pair's mix over the oracle's gated, rounded to FP32
        std::vector<float> g32(gated.begin(), gated.end());
        const float* d_g = dev.upload(g32);
        float* d_mref = dev.alloc<float>((size_t) (T * N));
        pf::gr_mix_r(d_R, d_rs, d_wn, d_g, d_mref, nullptr, T, par::stream());
        par::sync();
        par::copy_out(mixed.data(), d_mixed, mixed.size());
        par::copy_out(m16.data(), d_m16, m16.size());
        par::copy_out(m16lo.data(), d_m16lo, m16lo.size());
        par::copy_out(mh.data(), d_mh, mh.size());
        std::vector<float> mref((size_t) (T * N));
        par::copy_out(mref.data(), d_mref, mref.size());

        auto check_bound = [&](const std::vector<float>& out, double* worst_out) {
            int64_t over = 0;
            double worst = 0;
            for (int64_t i = 0; i < T * N; ++i) {
                const double e = std::fabs((double) out[(size_t) i] - mix_ref[(size_t) i]);
                if (!(e <= bound[(size_t) i])) ++over;
                if (bound[(size_t) i] > 0) worst = std::max(worst, e / bound[(size_t) i]);
            }
            if (worst_out) *worst_out = worst;
            return over;
        };
        double worst = 0;
        const int64_t over = check_bound(mixed, &worst);
        check(over == 0, "every output inside the bound against the float64 oracle");
        std::printf("  worst use of the bound: %.3g%%\n", 100.0 * worst);
        bool images = true, canary = true;
        double dref = 0, sref = 0;
        for (int64_t i = 0; i < T * N; ++i) {
            const float s = mixed[(size_t) i];
            const uint16_t h = bf16_from_f32(s);
            if (m16[(size_t) i] != h || m16lo[(size_t) i] != bf16_from_f32(s - f32_from_bf16(h)) ||
                mh[(size_t) i] != strata::kernels::f16_from_f32(s))
                images = false;
            dref = std::max(dref, (double) std::fabs(s - mref[(size_t) i]));
            sref = std::max(sref, (double) std::fabs(mref[(size_t) i]));
        }
        for (int64_t i = T * N; i < T * N + PAD; ++i)
            if (sycl::bit_cast<uint32_t>(mixed[(size_t) i]) != sycl::bit_cast<uint32_t>(CANARY) || m16[(size_t) i] != 0xABCD)
                canary = false;
        check(images, "BF16 (high, low) and FP16 images of mixed, bit for bit");
        check(canary, "outputs past T keep the canary");
        std::printf("  against gr_mix_r over the oracle's gated in FP32: largest difference %.3g (scale %.3g)\n", dref, sref);
        std::vector<float> moved = mixed;
        moved[(size_t) (T * N / 2)] = (float) (mix_ref[(size_t) (T * N / 2)] + 2.5 * bound[(size_t) (T * N / 2)]);
        check(check_bound(moved, nullptr) >= 1, "negative control: a 2.5B move fails the same check");
    }
    std::printf(g_fail == 0 ? "gr_upmix_parity: all cases PASS\n" : "gr_upmix_parity: %d FAILURES\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
