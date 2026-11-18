// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/spec/spec_width_law_test.cpp - does the cost rule's window (STRATA_SPEC_COST) keep a sampled request's
// distribution under rejection sampling (STRATA_SPEC_PROB's older rule, serve's set_spec_q)?
//
// Two rows, two tokens.  The drafter draws d1 from q1 (its top probability is the context's); the draft head then runs
// on d1, so the top probability of q2 depends on d1.  The window is DraftPolicy::keep_draft's, step by step, as the
// drafter asks it: best_rows (the widest the rule ever preferred) or prefix_rows (the drafts each kept at their own
// step).  Row 1 judges d1 with spec_verify_row (spec_prob.hpp, the host reference of the GPU's row) when the window
// offers it, and samples p1 when it does not.
//   1. exact enumeration: with costs where two rows do not pay and three do, best_rows offers d1 only when d1 leads
//      to a sure second draft, and the first token comes out (0.5625, 0.4375) for p1 = (0.5, 0.5); prefix_rows gives p1;
//   2. a window whose second draft depends on d1 (costs where two rows pay) keeps the joint law p1 x p2 with prefix_rows;
//   3. Monte Carlo, seeded: the first token's counts with drafts (best_rows of 1, prefix_rows of 2) against those
//      without, a chi-square on the 2 x 2 table.
#include "strata/core/spec_prob.hpp"
#include "strata/spec/draft_policy.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>

using strata::spec::DraftPolicy;
using Dist = std::array<double, 2>;
using QRow = std::array<float, 2>;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-74s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
bool near(const Dist& a, const Dist& b) { return std::abs(a[0] - b[0]) < 1e-9 && std::abs(a[1] - b[1]) < 1e-9; }

const int32_t kIds[2] = {0, 1};
// one verify row judging draft d against q, every branch of its acceptance draw (two tokens: the residual has one
// token, so the residual draw does not matter)
Dist judged(const Dist& p, const QRow& q, int d) {
    const strata::core::SpecP P{kIds, p.data(), 2};
    const strata::core::SpecQ Q{kIds, q.data(), 2};
    const double a = std::min(1.0, p[(size_t) d] / (double) q[(size_t) d]);
    Dist r{};
    if (a > 0.0) r[(size_t) strata::core::spec_verify_row(P, d, Q, (float) (a / 2), 0.5f).token] += a;
    if (a < 1.0) r[(size_t) strata::core::spec_verify_row(P, d, Q, (float) ((1 + a) / 2), 0.5f).token] += 1.0 - a;
    return r;
}
float top(const QRow& q) { return std::max(q[0], q[1]); }

// the drafter's chain as MtpDrafter runs it under STRATA_SPEC_COST: keep_draft after each draft, at most `cap` rows
struct Width {
    int best = 1, prefix = 1;
};
Width chain(const DraftPolicy& pol, const QRow& q1, const QRow& q2, int cap) {
    DraftPolicy::ChainState st;
    if (pol.keep_draft(st, 0, top(q1), cap)) (void) pol.keep_draft(st, 1, top(q2), cap);
    return {st.best_rows, st.prefix_rows};
}

DraftPolicy policy(double c1, double c2, double c3) {
    DraftPolicy pol(3);
    pol.set_robust(true);
    const double ms[3] = {c1, c2, c3};
    pol.seed_costs(ms, 3, 6.0);   // six rounds' worth each: no width is probed as unmeasured
    return pol;
}

const Dist kP1{0.5, 0.5};
const QRow kQ1{0.875f, 0.125f};
const Dist kP2[2] = {{0.3, 0.7}, {0.6, 0.4}};       // p2 given the first token
const QRow kQ2[2] = {{0.95f, 0.05f}, {0.45f, 0.55f}};   // q2 given d1: sure after token 0, unsure after token 1
}  // namespace

int main() {
    std::printf("spec_width_law_test\n");
    // 1. two rows cost more than they bring at q1's top alone; three pay when the second draft is sure
    const DraftPolicy look = policy(50.0, 95.0, 125.0);
    const Width w0 = chain(look, kQ1, kQ2[0], 3), w1 = chain(look, kQ1, kQ2[1], 3);
    check(w0.best == 3 && w1.best == 1, "setup: best_rows offers d1 only when d1 = 0 (a sure second draft)");
    check(w0.prefix == 1 && w1.prefix == 1, "setup: prefix_rows never offers d1 (two rows do not pay)");
    Dist best{}, prefix{};
    for (int d = 0; d < 2; ++d) {
        const Width w = d == 0 ? w0 : w1;
        const Dist rb = w.best >= 2 ? judged(kP1, kQ1, d) : kP1, rp = w.prefix >= 2 ? judged(kP1, kQ1, d) : kP1;
        for (int i = 0; i < 2; ++i) {
            best[(size_t) i] += kQ1[(size_t) d] * rb[(size_t) i];
            prefix[(size_t) i] += kQ1[(size_t) d] * rp[(size_t) i];
        }
    }
    check(near(best, Dist{0.5625, 0.4375}), "best_rows: the first token is (0.5625, 0.4375), not p1 (the older rule)");
    check(near(prefix, kP1), "prefix_rows: the first token follows p1");

    // 2. two rows pay; the third only after d1 = 0: row 2 is offered by d1, which is its own context
    const DraftPolicy pay = policy(50.0, 60.0, 80.0);
    const Width v0 = chain(pay, kQ1, kQ2[0], 3), v1 = chain(pay, kQ1, kQ2[1], 3);
    check(v0.prefix == 3 && v1.prefix == 2, "setup: prefix_rows offers d2 only after d1 = 0");
    double joint[2][2] = {};
    for (int d1 = 0; d1 < 2; ++d1)
        for (int d2 = 0; d2 < 2; ++d2) {
            const double w = kQ1[(size_t) d1] * kQ2[d1][(size_t) d2];
            const int T = d1 == 0 ? v0.prefix : v1.prefix;
            const Dist r1 = T >= 2 ? judged(kP1, kQ1, d1) : kP1;
            for (int o1 = 0; o1 < 2; ++o1) {
                // o1 == d1 only by acceptance (the residual never holds the draft); a rejected row ends the window
                // and the next one samples p2 given o1
                const bool on = T >= 3 && o1 == d1;
                const Dist r2 = on ? judged(kP2[o1], kQ2[d1], d2) : kP2[o1];
                for (int o2 = 0; o2 < 2; ++o2) joint[o1][o2] += w * r1[(size_t) o1] * r2[(size_t) o2];
            }
        }
    bool exact = true;
    for (int o1 = 0; o1 < 2; ++o1)
        for (int o2 = 0; o2 < 2; ++o2)
            // (q is float, as the drafter's rows are: 0.95f + 0.05f is 1 - 1.1e-8; a biased window is off by 0.06)
            exact = exact && std::abs(joint[o1][o2] - kP1[(size_t) o1] * kP2[o1][(size_t) o2]) < 1e-7;
    check(exact, "prefix_rows: the two tokens follow p1 x p2");

    // 3. the same window by sampling: token counts with drafts against counts without, chi-square on the 2 x 2 table
    //    (1 degree of freedom; 10.83 is p = 0.001).  Seeded, so the result does not change from run to run.
    std::mt19937_64 rng(0x5eedu);  // NOLINT(bugprone-random-generator-seed): reproducible counts.
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    const strata::core::SpecP P1{kIds, kP1.data(), 2};
    const strata::core::SpecQ Q1{kIds, kQ1.data(), 2};
    constexpr int kN = 200000;
    // the window of row 1 for each d1: best_rows of the first costs, prefix_rows of the second (it offers d1 always)
    const int rows_best[2] = {w0.best, w1.best}, rows_prefix[2] = {v0.prefix, v1.prefix}, rows_none[2] = {1, 1};
    auto count = [&](const int* rows) {
        std::array<double, 2> n{};
        for (int i = 0; i < kN; ++i) {
            const int d = uni(rng) < kQ1[0] ? 0 : 1;
            const float ua = uni(rng), ur = uni(rng), up = uni(rng);
            const int32_t o = rows[d] >= 2 ? strata::core::spec_verify_row(P1, d, Q1, ua, ur).token
                                           : strata::core::spec_sample(P1, up);
            n[(size_t) o] += 1.0;
        }
        return n;
    };
    auto chi2 = [](const std::array<double, 2>& a, const std::array<double, 2>& b) {
        const double na = a[0] + a[1], nb = b[0] + b[1], n = na + nb;
        double x = 0.0;
        for (int i = 0; i < 2; ++i) {
            const double col = a[(size_t) i] + b[(size_t) i];
            const double ea = na * col / n, eb = nb * col / n;
            x += (a[(size_t) i] - ea) * (a[(size_t) i] - ea) / ea + (b[(size_t) i] - eb) * (b[(size_t) i] - eb) / eb;
        }
        return x;
    };
    const auto plain = count(rows_none);
    const double x_best = chi2(count(rows_best), plain), x_prefix = chi2(count(rows_prefix), plain);
    std::printf("  chi-square against no drafts: best_rows %.1f, prefix_rows %.2f\n", x_best, x_prefix);
    check(x_best > 100.0, "best_rows: the counts differ from those without drafts");
    check(x_prefix < 10.83, "prefix_rows: the counts agree with those without drafts");

    std::printf("spec_width_law_test: %s\n", g_fail ? "FAILED" : "passed");
    return g_fail ? 1 : 0;
}
