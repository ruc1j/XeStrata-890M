// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/kernels/qsa_select_wmma_parity.cpp - the QSA block scores on WMMA (qsa_block_scores_wmma, STRATA_SELECT_WMMA=1 on
// gfx1150) against two references (GPU, synthetic, no model).  Transcribed from nicejunjie/Strata-rdna3's test of the
// same name (its matrix-core scorer came from xyzzing/Strata's RDNA work), on parity_device.hpp:
//
//   1. a float64 oracle: every score must lie inside the bound derived beforehand,
//          B(q, b) = 2^-9.5 * S(q, b) + 8 * 2^-24 * |score|,  S = the sum of the entry's absolute products;
//      Strata-rdna3 set it for FP16 operands with FP32 accumulation, and the bf16 x 3 scorer, FP32-level, keeps well
//      inside it (the worst use of the bound is printed);
//   2. the warp kernel (qsa_block_scores) against a host model of its arithmetic, a diagnostic only: which contraction
//      the compiler chose is reported, not required;
// then the tail block n_bid must be bitwise the warp kernel's (both use the same arithmetic), entries past n_bid keep
// a canary, rows past the reach are poisoned and must not reach a score, the selection ids (qsa_block_topk over both
// score arrays) must agree wherever the gap at the cut is wider than 2B (a narrower one is a legal near-tie, reported),
// and a negative control - the same bound check over a score array moved by 2.5B at one entry - must fail.
// Cases: ragged query tiles (nq % 16 != 0, 49 queries by default), ragged block tiles (reach % 16 != 0), whole tiles,
// an x8 magnitude, a degenerate n_kv = 0 query, and one-hot / coordinate-coded rows (every dot 0 or exact).
// Usage: qsa_select_wmma_parity [nq=49] [max_blocks=620]   (exit 0: all pass; 77: no WMMA scorer on this device)
//        qsa_select_wmma_parity --bench [context=32768] [queries=2048] [reps=10]: the prompt path's scorers timed
//        (the FP32 tiled one, qsa_block_scores_tc without the variable, and the WMMA one) on the last `queries`
//        positions of `context`
// The WMMA scorer is called directly (STRATA_SELECT_WMMA is qsa_block_scores_tc's switch).
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_select.hpp"

#include "parity_device.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

namespace k = strata::kernels;
namespace par = strata::parity;

namespace {

int g_fail = 0;
void check(bool ok, const char* what) {
    if (!ok) ++g_fail;
    std::printf("  %-66s %s\n", what, ok ? "ok" : "FAIL");
}

// The warp kernel's arithmetic for one (query, block) as lane 0 sees it: 32 lane partials of 4 products, then the
// xor butterfly 16, 8, 4, 2, 1 over the 32-lane array.  Six contraction shapes of the 4-product sum.
float warp_order_dot(const float* key, const float* q, int pattern) {
    float v[32];
    for (int l = 0; l < 32; ++l) {
        const float* k4 = key + l * 4;
        const float* q4 = q + l * 4;
        float d;
        if (pattern == 0) d = k4[0] * q4[0] + k4[1] * q4[1] + k4[2] * q4[2] + k4[3] * q4[3];
        else if (pattern == 1) {
            d = std::fma(k4[0], q4[0], k4[1] * q4[1]);
            d = std::fma(k4[2], q4[2], d);
            d = std::fma(k4[3], q4[3], d);
        } else if (pattern == 2) {
            d = std::fma(k4[0], q4[0], std::fma(k4[1], q4[1], std::fma(k4[2], q4[2], k4[3] * q4[3])));
        } else if (pattern == 3) {
            d = std::fma(k4[0], q4[0], k4[1] * q4[1]) + std::fma(k4[2], q4[2], k4[3] * q4[3]);
        } else if (pattern == 4) {
            d = std::fma(k4[0], q4[0], std::fma(k4[1], q4[1], k4[2] * q4[2])) + k4[3] * q4[3];
        } else {
            d = k4[0] * q4[0] + std::fma(k4[1], q4[1], std::fma(k4[2], q4[2], k4[3] * q4[3]));
        }
        v[l] = d;
    }
    for (int o = 16; o > 0; o >>= 1) {
        float t[32];
        for (int l = 0; l < 32; ++l) t[l] = v[l] + v[l ^ o];
        std::memcpy(v, t, sizeof v);
    }
    return v[0];
}

double oracle_score(const double* key, const double* q) {   // heads relu-summed, dots in float64
    double s = 0;
    for (int h = 0; h < 4; ++h) {
        double d = 0;
        for (int i = 0; i < 128; ++i) d += key[i] * q[h * 128 + i];
        s += d > 0 ? d : 0;
    }
    return s;
}

double abs_product_sum(const double* key, const double* q) {   // S(q, b) of the bound
    double s = 0;
    for (int h = 0; h < 4; ++h)
        for (int i = 0; i < 128; ++i) s += std::fabs(key[i] * q[h * 128 + i]);
    return s;
}

double bound_of(double S, double score) { return std::exp2(-9.5) * S + 8 * std::exp2(-24.0) * std::fabs(score); }

// --bench: the tiled FP32 scorer (qsa_block_scores_tc, run without STRATA_SELECT_WMMA) against the WMMA one, on random
// keys and queries, as the prompt path calls them (one call a chunk of queries)
int bench(int argc, char** argv) {
    const int64_t ctx = argc > 2 ? std::atoll(argv[2]) : 32768;
    const int64_t nq = argc > 3 ? std::atoll(argv[3]) : 2048;
    const int reps = argc > 4 ? std::atoi(argv[4]) : 10;
    const k::QsaShapes s = k::qsa_real_shapes();
    const int64_t max_blocks = ctx / 4 + 1;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> pooled((size_t) max_blocks * 128), dead(128), q((size_t) nq * 512);
    for (auto& x : pooled) x = nd(rng);
    for (auto& x : dead) x = nd(rng);
    for (auto& x : q) x = nd(rng);
    std::vector<int32_t> steps((size_t) nq * k::kStepCount, 0);
    for (int64_t i = 0; i < nq; ++i) {
        const int64_t n_kv = ctx - nq + i + 1;
        int32_t* st = &steps[(size_t) i * k::kStepCount];
        st[k::kStepPos] = (int32_t) (n_kv - 1);
        st[k::kStepNKv] = (int32_t) n_kv;
        st[k::kStepNBid] = (int32_t) (n_kv / 4);
        st[k::kStepWidth] = (int32_t) k::qsa_selection_width(n_kv, s);
    }
    const int64_t active = steps[(size_t) (nq - 1) * k::kStepCount + k::kStepNBid] + 1;
    par::Device dev;
    const float* d_pooled = dev.upload(pooled);
    const float* d_dead = dev.upload(dead);
    const float* d_q = dev.upload(q);
    const int32_t* d_steps = dev.upload(steps);
    float* d_a = dev.alloc<float>((size_t) nq * max_blocks);
    float* d_b = dev.alloc<float>((size_t) nq * max_blocks);
    auto timed = [&](auto f) {
        f();
        par::sync();
        const auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; ++r) f();
        par::sync();
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
    };
    // qsa_block_scores_tc reads STRATA_SELECT_WMMA once, at its first call: unset for it, so it times the tiled scorer;
    // the WMMA scorer reads the variable's value (2 or 3 pieces) at its own first call, after it is put back
    const char* sel_env = std::getenv("STRATA_SELECT_WMMA");
    const std::string sel_val = sel_env ? sel_env : "";
    unsetenv("STRATA_SELECT_WMMA");
    const double ms_simt = timed([&] {
        if (!k::qsa_block_scores_tc(d_pooled, d_dead, d_q, d_steps, nq, max_blocks, s, d_a, par::stream(), active))
            std::exit(2);
    });
    if (!sel_val.empty()) setenv("STRATA_SELECT_WMMA", sel_val.c_str(), 1);
    bool ok = true;
    const double ms_wmma = timed([&] {
        ok = ok && k::qsa_block_scores_wmma(d_pooled, d_dead, d_q, d_steps, nq, max_blocks, s, d_b, par::stream(), active);
    });
    if (!ok) {
        std::printf("SKIP: no WMMA block scorer on this device (gfx1150 only)\n");
        return 77;
    }
    std::vector<float> a((size_t) nq * max_blocks), b(a.size());
    par::copy_out(a.data(), d_a, a.size());
    par::copy_out(b.data(), d_b, b.size());
    double md = 0, ms = 0;
    for (int64_t i = 0; i < nq; ++i)
        for (int64_t blk = 0; blk < steps[(size_t) i * k::kStepCount + k::kStepNBid]; ++blk) {
            const size_t at = (size_t) (i * max_blocks + blk);
            md = std::max(md, (double) std::fabs(a[at] - b[at]));
            ms = std::max(ms, (double) std::fabs(a[at]));
        }
    std::printf("context %lld, %lld queries: FP32 tiled %.3f ms, WMMA %.3f ms (%.2fx); largest difference %.3g (scale %.3g)\n",
                (long long) ctx, (long long) nq, ms_simt, ms_wmma, ms_simt / ms_wmma, md, ms);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--bench") return bench(argc, argv);
    const bool selftest = argc > 1 && std::string(argv[1]) == "--selftest";
    const int64_t nq0 = (argc > 1 && !selftest) ? std::atoll(argv[1]) : 49;
    const int64_t max_blocks = (argc > 2 && !selftest) ? std::atoll(argv[2]) : 620;
    const k::QsaShapes s = k::qsa_real_shapes();

    // pattern 0: a shared direction plus noise (a spread like a real indexer's); pattern 1: one-hot and coordinate-coded
    // rows, every expected dot 0 or exact, so a lane, transpose or store error cannot hide inside the tolerance
    struct Case { const char* name; int64_t nq; int64_t reach; float scale; int64_t nkv0; int pattern; };
    const Case cases[] = {
        {"ragged query tiles (nq % 16 != 0) + ragged block tiles", nq0, max_blocks - 3, 1.0f, 0, 0},
        {"whole tiles (nq % 16 == 0, reach % 16 == 0)", (nq0 / 16) * 16, ((max_blocks - 3) / 16) * 16, 1.0f, 0, 0},
        {"x8 magnitude", nq0, max_blocks - 3, 8.0f, 0, 0},
        {"degenerate step n_kv = 0 on the first query", nq0, max_blocks - 3, 1.0f, 1, 0},
        {"one-hot + coordinate-coded rows (lane / store adversarial)", nq0, max_blocks - 3, 1.0f, 0, 1},
    };

    par::Device dev;
    for (const Case& cs : cases) {
        std::printf("case: %s (%lld queries, reach %lld)\n", cs.name, (long long) cs.nq, (long long) cs.reach);
        const int64_t nq = cs.nq, reach = cs.reach;
        if (nq < 1 || reach < 4) continue;
        std::mt19937 rng(1234);
        std::normal_distribution<float> nd(0.f, 1.f);
        std::vector<float> dir(128);
        for (auto& x : dir) x = nd(rng);
        // pooled keys; rows past `reach` POISONED - the scorer must never let them reach a score
        std::vector<float> pooled((size_t) max_blocks * 128), dead(128), q((size_t) nq * 512);
        for (int64_t b = 0; b < max_blocks; ++b) {
            const float a = nd(rng);
            float* row = &pooled[(size_t) b * 128];
            for (int d = 0; d < 128; ++d) {
                if (cs.pattern == 1) row[d] = b < reach ? (d == (b % 128) ? 1.0f : 0.0f) : 1e30f;
                else row[d] = b < reach ? (0.5f * a * dir[d] + nd(rng)) * cs.scale : 1e30f;
            }
        }
        for (int d = 0; d < 128; ++d) dead[d] = cs.pattern == 1 ? (d == 61 ? 1.0f : 0.0f) : nd(rng) * cs.scale;
        for (int64_t i = 0; i < nq; ++i)
            for (int d = 0; d < 512; ++d) {
                const int h = d / 128, di = d % 128;
                q[(size_t) i * 512 + d] =
                    cs.pattern == 1
                        ? (di == ((i * 4 + h) * 37 % 128) ? 1.0f : (di == ((i * 7 + h) * 53 % 128) ? -0.25f : 0.0f))
                        : (0.2f * dir[d % 128] + 0.1f * nd(rng)) * cs.scale;
            }
        // steps: mixed n_bid (neighbours differ), optionally a degenerate n_kv = 0 first query
        std::vector<int32_t> steps((size_t) nq * k::kStepCount, 0);
        for (int64_t i = 0; i < nq; ++i) {
            int32_t* st = &steps[(size_t) i * k::kStepCount];
            const int64_t n_bid = (i == 0 && cs.nkv0) ? 0 : 1 + (i * 7919) % (reach - 1);
            const int64_t n_kv = (i == 0 && cs.nkv0) ? 0 : n_bid * 4 + 1 + (int) (i % 3);
            st[k::kStepPos] = (int32_t) (n_kv - 1);
            st[k::kStepNKv] = (int32_t) n_kv;
            st[k::kStepNBid] = (int32_t) n_bid;
            st[k::kStepWidth] = (int32_t) k::qsa_selection_width(n_kv, s);
        }
        const float CANARY = -12345.5f;
        std::vector<float> sc_warp((size_t) nq * max_blocks, CANARY), sc_wmma((size_t) nq * max_blocks, CANARY);

        const float* d_pooled = dev.upload(pooled);
        const float* d_dead = dev.upload(dead);
        const float* d_q = dev.upload(q);
        const int32_t* d_steps = dev.upload(steps);
        float* d_warp = dev.upload(sc_warp);
        float* d_wmma = dev.upload(sc_wmma);

        k::qsa_block_scores(d_pooled, d_dead, d_q, d_steps, nq, max_blocks, s, d_warp, par::stream(), reach);
        if (!k::qsa_block_scores_wmma(d_pooled, d_dead, d_q, d_steps, nq, max_blocks, s, d_wmma, par::stream(),
                                      reach)) {
            std::printf("SKIP: no WMMA block scorer on this device (gfx1150 only)\n");
            return 77;
        }
        par::sync();
        par::copy_out(sc_warp.data(), d_warp, sc_warp.size());
        par::copy_out(sc_wmma.data(), d_wmma, sc_wmma.size());

        // 1. the warp kernel against its host model (diagnostic): how many entries each contraction shape fits
        int64_t fit[6] = {0, 0, 0, 0, 0, 0}, total = 0;
        for (int64_t i = 0; i < nq; ++i) {
            const int32_t* st = &steps[(size_t) i * k::kStepCount];
            const int64_t n_kv = st[k::kStepNKv], n_bid = st[k::kStepNBid];
            for (int64_t b = 0; b <= n_bid && b < max_blocks; ++b) {
                const float* key = (b == n_bid) ? dead.data() : &pooled[(size_t) b * 128];
                const float got = sc_warp[(size_t) i * max_blocks + b];
                ++total;
                for (int p = 0; p < 6; ++p) {
                    float score = 0;
                    for (int h = 0; h < 4; ++h)
                        score += std::fmax(warp_order_dot(key, &q[(size_t) i * 512 + h * 128], p), 0.0f);
                    if (b == n_bid && n_kv % 4 != 0) score += 1e9f;
                    if (std::memcmp(&got, &score, 4) == 0) ++fit[p];
                }
            }
        }
        std::printf("  warp kernel's contraction (diagnostic): plain %lld, fma-left %lld, fma-right %lld, two-fma %lld, "
                    "left3+mul %lld, mul+right3 %lld of %lld\n",
                    (long long) fit[0], (long long) fit[1], (long long) fit[2], (long long) fit[3], (long long) fit[4],
                    (long long) fit[5], (long long) total);

        // 2. the WMMA scores against the float64 oracle, inside the bound; the tail bitwise the warp kernel's.  The
        // bound check is a lambda so that the negative control runs the same code
        auto check_bound = [&](const std::vector<float>& scores, double* worst_out) {
            int64_t violations = 0;
            double worst = 0;
            std::vector<double> kd(128), qd(512);
            for (int64_t i = 0; i < nq; ++i) {
                const int64_t n_bid = steps[(size_t) i * k::kStepCount + k::kStepNBid];
                for (int d = 0; d < 512; ++d) qd[d] = q[(size_t) i * 512 + d];
                for (int64_t b = 0; b < n_bid && b < max_blocks; ++b) {
                    for (int d = 0; d < 128; ++d) kd[d] = pooled[(size_t) b * 128 + d];
                    const double o = oracle_score(kd.data(), qd.data());
                    const double B = bound_of(abs_product_sum(kd.data(), qd.data()), o);
                    const double err = std::fabs((double) scores[(size_t) i * max_blocks + b] - o);
                    if (!(err <= B)) ++violations;
                    if (B > 0 && err / B > worst) worst = err / B;
                }
            }
            if (worst_out) *worst_out = worst;
            return violations;
        };
        bool tail_same = true, untouched = true;
        for (int64_t i = 0; i < nq; ++i) {
            const int64_t n_bid = steps[(size_t) i * k::kStepCount + k::kStepNBid];
            for (int64_t b = 0; b < max_blocks; ++b) {
                const float got = sc_wmma[(size_t) i * max_blocks + b];
                if (b > n_bid) {
                    if (std::memcmp(&got, &CANARY, 4) != 0) untouched = false;
                    continue;
                }
                if (b == n_bid) {
                    const float w = sc_warp[(size_t) i * max_blocks + b];
                    if (std::memcmp(&got, &w, 4) != 0) tail_same = false;
                }
            }
        }
        double worst = 0;
        const int64_t violations = check_bound(sc_wmma, &worst);
        check(untouched, "entries past n_bid keep the canary (as the warp kernel leaves them)");
        check(tail_same, "tail block bitwise the warp kernel's (the same tail arithmetic)");
        check(violations == 0, "every score inside the bound against the float64 oracle");
        std::printf("  worst use of the bound: %.3g%%\n", 100.0 * worst);

        // 3. selection ids: the same wherever the gap at the cut is wider than 2B; a narrower one only reported
        const int64_t cap = k::qsa_selection_width(k::kTopkMaxCells, s);
        std::vector<int32_t> ids_warp((size_t) nq * cap), ids_wmma((size_t) nq * cap);
        int32_t* d_iw = dev.alloc<int32_t>(ids_warp.size());
        int32_t* d_im = dev.alloc<int32_t>(ids_wmma.size());
        k::qsa_block_topk(d_warp, d_steps, nq, max_blocks, cap, s, d_iw, par::stream());
        k::qsa_block_topk(d_wmma, d_steps, nq, max_blocks, cap, s, d_im, par::stream());
        par::sync();
        par::copy_out(ids_warp.data(), d_iw, ids_warp.size());
        par::copy_out(ids_wmma.data(), d_im, ids_wmma.size());
        int64_t diff_queries = 0, illegal = 0, near_ties = 0;
        for (int64_t i = 0; i < nq; ++i) {
            const int w = steps[(size_t) i * k::kStepCount + k::kStepWidth];
            if (std::memcmp(&ids_warp[(size_t) i * cap], &ids_wmma[(size_t) i * cap], (size_t) w * 4) == 0) continue;
            ++diff_queries;
            // the cut in this query's blocks below n_bid (the tail block, its score raised by 1e9 when partial, is
            // taken first): the gap between the blocks either side of it, in the warp kernel's scores, against the
            // bound of those blocks
            const int32_t* st = &steps[(size_t) i * k::kStepCount];
            const int64_t n_kv = st[k::kStepNKv], n_bid = st[k::kStepNBid];
            const int64_t nb = std::min<int64_t>(n_bid, max_blocks);
            std::vector<int64_t> order((size_t) nb);
            for (int64_t b = 0; b < nb; ++b) order[(size_t) b] = b;
            const float* wrow = &sc_warp[(size_t) i * max_blocks];
            std::stable_sort(order.begin(), order.end(), [&](int64_t x, int64_t y) { return wrow[x] > wrow[y]; });
            const int64_t kb = (w - (n_kv - n_bid * 4)) / 4;   // blocks taken whole
            std::vector<double> kd(128), qd(512);
            for (int d = 0; d < 512; ++d) qd[d] = q[(size_t) i * 512 + d];
            auto bound_at = [&](int64_t b) {
                for (int d = 0; d < 128; ++d) kd[d] = pooled[(size_t) b * 128 + d];
                return bound_of(abs_product_sum(kd.data(), qd.data()), oracle_score(kd.data(), qd.data()));
            };
            bool tie = false;
            for (int64_t j = std::max<int64_t>(kb - 1, 0); j + 1 < nb && j <= kb; ++j) {
                const int64_t x = order[(size_t) j], y = order[(size_t) j + 1];
                const double gap = (double) wrow[x] - (double) wrow[y];
                if (gap <= 2 * std::max(bound_at(x), bound_at(y))) tie = true;
            }
            if (tie) ++near_ties;
            else ++illegal;
        }
        check(illegal == 0, "selection ids agree wherever the gap exceeds 2B");
        std::printf("  differing queries: %lld (legal near-ties within 2B: %lld)\n", (long long) diff_queries,
                    (long long) near_ties);

        // 4. negative control: the same bound check over an array moved by 2.5B at one entry must fail
        std::vector<float> perturbed = sc_wmma;
        {
            int64_t pi = -1, pb = -1;
            for (int64_t i = 0; i < nq && pi < 0; ++i)
                if (steps[(size_t) i * k::kStepCount + k::kStepNBid] > 0) { pi = i; pb = 0; }
            if (pi >= 0) {
                std::vector<double> kd(128), qd(512);
                for (int d = 0; d < 128; ++d) kd[d] = pooled[(size_t) pb * 128 + d];
                for (int d = 0; d < 512; ++d) qd[d] = q[(size_t) pi * 512 + d];
                const double o = oracle_score(kd.data(), qd.data());
                const double B = bound_of(abs_product_sum(kd.data(), qd.data()), o);
                perturbed[(size_t) pi * max_blocks + pb] = (float) (o + 2.5 * B);
            }
        }
        check(check_bound(perturbed, nullptr) >= 1, "negative control: a 2.5B move fails the same check");
    }

    std::printf(g_fail == 0 ? "qsa_select_wmma_parity: all cases PASS\n" : "qsa_select_wmma_parity: %d FAILURES\n",
                g_fail);
    return g_fail == 0 ? 0 : 1;
}
