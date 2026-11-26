// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/dflash2_attention_parity.cpp - the DFlash2 drafter's attention kernels against a float64 reference (GPU,
// synthetic, no model): the FP32 kernels (STRATA_DF2_ATTN=1, the 8-query-group form; 2, upstream's 256-work-item
// form; 4, the same on 2 x 2 register tiles, the default on an integrated GPU on the HIP backend such as the 890M; 5,
// the 128-work-item form on 4 x 4 tiles) and the one on RDNA's matrix instructions (STRATA_DF2_ATTN=3, FP16 operands).
//
// The kernels leave partials (a running maximum, a sum and 128 output sums for each chunk of keys, K/V head and query);
// the test combines them as the merge kernel does and compares the 32 queries x 8 K/V heads x 128 outputs with
//     out = softmax(Q K^T / sqrt(128)) V
// computed in float64 over the keys each row may see.  The error is measured against the output's RMS: the FP32
// kernels must stay within 1e-4 of it and the FP16 one within 1e-2 (FP16's 2^-11 a factor, through the scores'
// exponentials); the worst use of each bound is printed.  Cases: a full window, a short context, no context, a window that masks the
// oldest keys of the later rows, and ring slots a tag marks as stale.  A negative control - one key moved in the
// reference only - must fail the FP16 bound.
// Usage: dflash2_attention_parity   (exit 0: all pass; 77: this device has no such instructions, only the FP32 kernels
// ran)
#include "strata/kernels/dflash2.hpp"

#include "parity_device.hpp"
#include "xe/amd_wmma.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <vector>

namespace k = strata::kernels;
namespace par = strata::parity;

namespace {

constexpr int HD = k::kDf2Hd, KV = k::kDf2Kv, ROWS = k::kDf2Rows, GQA = k::kDf2Gqa;
constexpr int NQ = ROWS * GQA;                         // queries of a K/V head: 4 heads x 8 rows
constexpr size_t PART = (size_t) k::kDf2PartChunks * KV * NQ * 130;

int g_fail = 0;
void check(bool ok, const char* what) {
    if (!ok) ++g_fail;
    std::printf("  %-70s %s\n", what, ok ? "ok" : "FAIL");
}

// a ring slot's tag: the epoch in the upper half, the position it holds in the lower
int64_t tag_of(int epoch, int pos) { return (int64_t) epoch * (int64_t) 4294967296 + (int64_t) pos; }

struct Case {
    const char* name;
    int s, window, ring;
    bool tags;
};

// softmax(Q K^T / sqrt(HD)) V in float64 for every (K/V head, query), over the keys the query's row may see
std::vector<double> reference(const Case& c, const std::vector<float>& q, const std::vector<float>& kring, const std::vector<float>& vring,
                              const std::vector<float>& kblk, const std::vector<float>& vblk, const std::vector<int64_t>& tag, int epoch) {
    const int lo = std::max(0, c.s - (c.window - 1)), nctx = c.s - lo, nkeys = nctx + ROWS;
    std::vector<double> out((size_t) KV * NQ * HD, 0.0);
    std::vector<double> sc((size_t) nkeys);
    for (int kvh = 0; kvh < KV; ++kvh)
        for (int qi = 0; qi < NQ; ++qi) {
            const int row = qi / 4, g = qi % 4;
            const float* qv = &q[((size_t) row * 32 + (size_t) kvh * GQA + g) * HD];
            double m = -std::numeric_limits<double>::infinity();
            for (int kk = 0; kk < nkeys; ++kk) {
                bool ok = true;
                const float* key;
                if (kk < nctx) {
                    const int j = lo + kk;
                    ok = j > c.s + row - c.window;
                    if (ok && c.tags) ok = tag[(size_t) (j % c.ring)] == tag_of(epoch, j);
                    key = &kring[((size_t) kvh * c.ring + (size_t) (j % c.ring)) * HD];
                } else {
                    key = &kblk[((size_t) (kk - nctx) * KV + kvh) * HD];
                }
                double d = 0;
                for (int i = 0; i < HD; ++i) d += (double) qv[i] * (double) key[i];
                sc[(size_t) kk] = ok ? d / std::sqrt((double) HD) : -std::numeric_limits<double>::infinity();
                m = std::max(m, sc[(size_t) kk]);
            }
            double l = 0;
            double* o = &out[((size_t) kvh * NQ + qi) * HD];
            for (int kk = 0; kk < nkeys; ++kk) {
                if (std::isinf(sc[(size_t) kk])) continue;
                const double p = std::exp(sc[(size_t) kk] - m);
                const float* v = kk < nctx ? &vring[((size_t) kvh * c.ring + (size_t) ((lo + kk) % c.ring)) * HD]
                                           : &vblk[((size_t) (kk - nctx) * KV + kvh) * HD];
                l += p;
                for (int i = 0; i < HD; ++i) o[i] += p * (double) v[i];
            }
            for (int i = 0; i < HD; ++i) o[i] /= l;
        }
    return out;
}

// the kernels' partials combined as the merge kernel combines them
std::vector<double> combine(const std::vector<float>& part, int nch) {
    std::vector<double> out((size_t) KV * NQ * HD, 0.0);
    for (int kvh = 0; kvh < KV; ++kvh)
        for (int qi = 0; qi < NQ; ++qi) {
            double m = -std::numeric_limits<double>::infinity();
            for (int ch = 0; ch < nch; ++ch) m = std::max(m, (double) part[(((size_t) ch * KV + kvh) * NQ + qi) * 130]);
            double l = 0;
            double* o = &out[((size_t) kvh * NQ + qi) * HD];
            for (int ch = 0; ch < nch; ++ch) {
                const float* p = &part[(((size_t) ch * KV + kvh) * NQ + qi) * 130];
                if (std::isinf(p[0])) continue;
                const double w = std::exp((double) p[0] - m);
                l += (double) p[1] * w;
                for (int i = 0; i < HD; ++i) o[i] += (double) p[2 + i] * w;
            }
            for (int i = 0; i < HD; ++i) o[i] /= l;
        }
    return out;
}

// the worst |got - want| over the outputs, as a fraction of the reference's RMS
double worst(const std::vector<double>& got, const std::vector<double>& want) {
    double ss = 0, mx = 0;
    for (size_t i = 0; i < want.size(); ++i) {
        ss += want[i] * want[i];
        const double e = std::fabs(got[i] - want[i]);
        mx = std::isnan(e) ? std::numeric_limits<double>::infinity() : std::max(mx, e);
    }
    return mx / std::sqrt(ss / (double) want.size());
}

}  // namespace

int main() {
    auto& rt = strata::core::Runtime::get();
    const bool wmma = strata::kernels::xe::amd_wmma(rt.compute().get_device());
    constexpr double kTol32 = 1e-4, kTol16 = 1e-2;
    const Case cases[] = {
        {"a full window (2,047 context keys)", 5000, 2048, 2048, false},
        {"a short context (37 keys)", 37, 2048, 2048, false},
        {"no context (the block's keys only)", 0, 2048, 2048, false},
        {"a window of 300: the later rows lose the oldest keys", 3000, 300, 2048, false},
        {"stale ring slots (tags)", 4100, 2048, 2048, true},
    };
    std::mt19937 rng(11);   // NOLINT(bugprone-random-generator-seed): the same data each run.
    std::normal_distribution<float> nd(0.f, 1.f);
    par::Device dev;
    for (const Case& c : cases) {
        std::printf("%s (s = %d, window = %d)\n", c.name, c.s, c.window);
        std::vector<float> q((size_t) ROWS * 32 * HD), kring((size_t) KV * c.ring * HD), vring(kring.size());
        std::vector<float> kblk((size_t) ROWS * KV * HD), vblk(kblk.size());
        for (auto* v : {&q, &kring, &vring, &kblk, &vblk})
            for (auto& x : *v) x = nd(rng);
        const int epoch = 3;
        std::vector<int64_t> tag((size_t) c.ring);
        const int lo = std::max(0, c.s - (c.window - 1));
        for (int j = lo; j < c.s; ++j) tag[(size_t) (j % c.ring)] = tag_of(epoch, j);
        if (c.tags)   // every seventh slot holds another position, three more another epoch
            for (int j = lo; j < c.s; ++j) {
                if (j % 7 == 3) tag[(size_t) (j % c.ring)] = tag_of(epoch, j + c.ring);
                if (j % 501 == 9) tag[(size_t) (j % c.ring)] = tag_of(epoch - 1, j);
            }
        const std::vector<int32_t> anchor{c.s}, ep{epoch};
        float* d_q = dev.upload(q);
        float* d_kr = dev.upload(kring);
        float* d_vr = dev.upload(vring);
        float* d_kb = dev.upload(kblk);
        float* d_vb = dev.upload(vblk);
        int32_t* d_s = dev.upload(anchor);
        int32_t* d_ep = dev.upload(ep);
        int64_t* d_tag = dev.upload(tag);
        float* d_part = dev.alloc<float>(PART);
        int8_t* d_xq = dev.alloc<int8_t>((size_t) ROWS * 32 * HD);
        float* d_xd = dev.alloc<float>((size_t) ROWS * 32 * HD / 32);
        const std::vector<double> want = reference(c, q, kring, vring, kblk, vblk, tag, epoch);
        std::vector<float> part(PART);
        for (int variant : {1, 2, 4, 5, 3}) {
            if (variant == 3 && !wmma) continue;
            std::fill(part.begin(), part.end(), std::numeric_limits<float>::quiet_NaN());
            par::copy_in(d_part, part.data(), part.size());
            k::df2_attention_variant(variant);
            k::df2_attention(d_q, d_kr, d_vr, d_kb, d_vb, d_s, c.ring, c.window, d_part, d_xq, d_xd, par::stream(), c.tags ? d_tag : nullptr,
                             c.tags ? d_ep : nullptr);
            par::sync();
            par::copy_out(part.data(), d_part, part.size());
            const double e = worst(combine(part, k::kDf2PartChunks), want);
            char what[160];
            std::snprintf(what, sizeof what, "STRATA_DF2_ATTN=%d: worst error %.3g of the RMS (bound %.0e, %.1f%% used)", variant, e,
                          variant == 3 ? kTol16 : kTol32, 100.0 * e / (variant == 3 ? kTol16 : kTol32));
            check(e <= (variant == 3 ? kTol16 : kTol32), what);
            if (variant == 3 && &c == &cases[0]) {   // the negative control: the reference with one key of K/V head 0 moved
                std::vector<float> moved = kring;
                const int j = c.s - 5;
                for (int i = 0; i < HD; ++i) moved[((size_t) (j % c.ring)) * HD + (size_t) i] += 0.5f * q[(size_t) i];
                const double en = worst(combine(part, k::kDf2PartChunks), reference(c, q, moved, vring, kblk, vblk, tag, epoch));
                std::snprintf(what, sizeof what, "the negative control is detected (%.3g of the RMS)", en);
                check(en > kTol16, what);
            }
        }
    }
    k::df2_attention_variant(0);
    if (g_fail != 0) {
        std::printf("dflash2_attention_parity: %d FAILED\n", g_fail);
        return 1;
    }
    if (!wmma) {
        std::printf("dflash2_attention_parity: the FP32 kernels pass; this device has no WMMA, STRATA_DF2_ATTN=3 did not run\n");
        return 77;
    }
    std::printf("dflash2_attention_parity: ALL PASS\n");
    return 0;
}
