// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/qsa_score_parity.cpp - the indexer's block scores as the decode window computes them.
//
//   * qsa_block_scores (a sub-group a query and block) and qsa_block_scores_rows (a sub-group a block, every row of
//     the window) against a host reference in FP64: for a row with n_bid completed blocks, block b < n_bid scores
//     sum over the indexer's heads of relu(<key b, q head>), block n_bid scores the dead key the same way, plus 1e9
//     when the row has cells in it (n_kv % 4 != 0).  Within a relative 1e-4 (the kernels sum in FP32, by lanes).
//   * the rows form against the per-row form, memcmp (the same pairings, so the same bits);
//   * a block past a row's n_bid is not written;
//   * the rows form writes nothing while the last row selects every cell (n_kv <= width).
// The launches take the grid a window graph takes (no reach given): contexts below, at and above one sub-group a
// block, and max_blocks far above the context.
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_select.hpp"

#include "parity_device.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

namespace P = strata::parity;

template <typename T> T* up(const std::vector<T>& h) {
    T* p = static_cast<T*>(P::alloc_bytes(h.size() * sizeof(T) + 64));
    P::copy_bytes_in(p, h.data(), h.size() * sizeof(T));
    return p;
}

}  // namespace

int main() {
    using namespace strata::kernels;
    std::mt19937 rng(2051);  // NOLINT(bugprone-random-generator-seed): reproducible fixtures.
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    const QsaShapes s = qsa_real_shapes();
    const int64_t D = s.idx_dim, H = s.idx_n_head, R = s.idx_block;
    void* st = P::stream();
    int bad = 0, cases = 0;
    const float kUnwritten = -12345.0f;

    for (const int64_t max_blocks : {(int64_t) 1026, (int64_t) 8194, (int64_t) 65538}) {
        std::vector<float> pooled((size_t) (max_blocks * D)), dead((size_t) D);
        for (auto& v : pooled) v = gauss(rng);
        for (auto& v : dead) v = gauss(rng);
        const float* d_pooled = up(pooled);
        const float* d_dead = up(dead);
        // the first row's cell count: no completed block, a few, a sub-group's worth around the fixed grid, more
        for (const int64_t n_kv0 : {(int64_t) 1, (int64_t) 4, (int64_t) 29, (int64_t) 32, (int64_t) 2044, (int64_t) 2048,
                                    (int64_t) 2051, (int64_t) 4097, (max_blocks - 2) * R - 5}) {
            for (const int nq : {1, 2, 3, 8}) {
                if (n_kv0 + nq - 1 > (max_blocks - 1) * R) continue;
                std::vector<int32_t> steps((size_t) nq * kStepCount);
                for (int qi = 0; qi < nq; ++qi) {   // consecutive positions of one sequence
                    const int64_t n_kv = n_kv0 + qi;
                    steps[(size_t) qi * kStepCount + kStepPos] = (int32_t) (n_kv - 1);
                    steps[(size_t) qi * kStepCount + kStepNKv] = (int32_t) n_kv;
                    steps[(size_t) qi * kStepCount + kStepNBid] = (int32_t) (n_kv / R);
                    // a width below the cells: the selection is not every cell, so the rows form scores (the real
                    // width is every cell up to 2,051; that case is checked apart, below)
                    steps[(size_t) qi * kStepCount + kStepWidth] = (int32_t) (n_kv - 1);
                }
                std::vector<float> q((size_t) (nq * H * D));
                for (auto& v : q) v = gauss(rng);
                const int32_t* d_steps = up(steps);
                const float* d_q = up(q);
                const std::vector<float> fill((size_t) (nq * max_blocks), kUnwritten);
                float* d_one = up(fill);
                float* d_rows = up(fill);
                qsa_block_scores(d_pooled, d_dead, d_q, d_steps, nq, max_blocks, s, d_one, st);
                qsa_block_scores_rows(d_pooled, d_dead, d_q, d_steps, nq, max_blocks, s, d_rows, st);
                P::sync();
                std::vector<float> one(fill.size()), rows(fill.size());
                P::copy_bytes_out(one.data(), d_one, one.size() * sizeof(float));
                P::copy_bytes_out(rows.data(), d_rows, rows.size() * sizeof(float));
                int case_bad = 0;
                if (std::memcmp(one.data(), rows.data(), one.size() * sizeof(float)) != 0) {
                    std::printf("    *** the rows form differs from the per-row form: max_blocks %lld, n_kv %lld, %d rows ***\n",
                                (long long) max_blocks, (long long) n_kv0, nq);
                    ++case_bad;
                }
                for (int qi = 0; qi < nq && case_bad == 0; ++qi) {
                    const int64_t n_kv = n_kv0 + qi, n_bid = n_kv / R;
                    for (int64_t b = 0; b < max_blocks; ++b) {
                        const float got = one[(size_t) (qi * max_blocks + b)];
                        if (b > n_bid) {
                            if (got != kUnwritten) {
                                std::printf("    *** block %lld past n_bid %lld was written ***\n", (long long) b, (long long) n_bid);
                                ++case_bad;
                                break;
                            }
                            continue;
                        }
                        const float* key = b == n_bid ? dead.data() : pooled.data() + (size_t) (b * D);
                        double want = 0.0, mag = 0.0;
                        for (int64_t h = 0; h < H; ++h) {
                            double dot = 0.0, abs_dot = 0.0;
                            const float* qh = q.data() + (size_t) ((qi * H + h) * D);
                            for (int64_t i = 0; i < D; ++i) {
                                dot += (double) key[i] * (double) qh[i];
                                abs_dot += std::fabs((double) key[i] * (double) qh[i]);
                            }
                            want += dot > 0.0 ? dot : 0.0;
                            mag += abs_dot;
                        }
                        if (b == n_bid && n_kv % R != 0) want += 1e9;
                        // a head whose dot is near zero may fall on the other side of the relu in FP32: the bound is
                        // on the products' magnitude, not on the score
                        const double tol = b == n_bid && n_kv % R != 0 ? 128.0 : 1e-4 * mag + 1e-5;
                        if (!(std::fabs((double) got - want) <= tol)) {
                            std::printf("    *** max_blocks %lld, n_kv %lld, row %d, block %lld: %g, the reference %g ***\n",
                                        (long long) max_blocks, (long long) n_kv0, qi, (long long) b, (double) got, want);
                            ++case_bad;
                            break;
                        }
                    }
                }
                // the last row selects every cell (n_kv <= width): the rows form writes nothing
                if (case_bad == 0) {
                    std::vector<int32_t> all = steps;
                    for (int qi = 0; qi < nq; ++qi)
                        all[(size_t) qi * kStepCount + kStepWidth] = all[(size_t) qi * kStepCount + kStepNKv];
                    const int32_t* d_all = up(all);
                    float* d_skip = up(fill);
                    qsa_block_scores_rows(d_pooled, d_dead, d_q, d_all, nq, max_blocks, s, d_skip, st);
                    P::sync();
                    std::vector<float> skipped(fill.size());
                    P::copy_bytes_out(skipped.data(), d_skip, skipped.size() * sizeof(float));
                    if (std::memcmp(skipped.data(), fill.data(), fill.size() * sizeof(float)) != 0) {
                        std::printf("    *** the rows form wrote scores although every cell is selected ***\n");
                        ++case_bad;
                    }
                }
                ++cases;
                bad += case_bad;
            }
        }
        std::printf("  block scores, max_blocks %-6lld %s\n", (long long) max_blocks, bad ? "*** NO ***" : "match");
    }
    std::printf("\nqsa_score: %d failures in %d cases\n", bad, cases);
    if (bad) return 1;
    std::printf("qsa_score_parity OK\n");
    return 0;
}
