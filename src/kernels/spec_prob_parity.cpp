// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/spec_prob_parity.cpp - STRATA_SPEC_PROB's verifier (sample_tokens_spec, core/spec_prob.hpp) keeps the
// target's distribution: whatever the drafter's q, every judged row's token follows p, the chain's distribution.
//
// A window of T rows over the same logits: row 0's draft is drawn from a q list that differs from p (it moves mass
// to the tail and puts some on a token p never emits), row 1's is a point mass (no list), row T - 1 has no draft
// (the plain sample).  Each trial draws its drafts on the host and runs the window at its own Philox counter; the
// rows' token counts are compared with p by chi-square.  Controls: the drafts themselves (distributed as q) must
// fail the same test, and p from the plain sampler (sample_tokens) must pass it.
#include "strata/kernels/sampler.hpp"
#include "strata/core/spec_prob.hpp"

#include "parity_device.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <random>
#include <vector>

namespace {

using strata::kernels::SamplerParams;

// the chain's kept list and its probabilities (top_k by value then id, top_p, temperature), as the sampler builds it
std::vector<std::pair<int, double>> chain(const std::vector<float>& l, const SamplerParams& p) {
    std::vector<int> ids(l.size());
    for (size_t i = 0; i < ids.size(); ++i) ids[i] = (int) i;
    std::stable_sort(ids.begin(), ids.end(), [&](int a, int b) { return l[(size_t) a] > l[(size_t) b]; });
    ids.resize((size_t) p.top_k);
    const double mx = l[(size_t) ids[0]];
    double sum = 0;
    for (int id : ids) sum += std::exp(l[(size_t) id] - mx);
    size_t cut = ids.size();
    double cum = 0;
    for (size_t i = 0; i < ids.size(); ++i) {
        cum += std::exp(l[(size_t) ids[i]] - mx) / sum;
        if (cum >= p.top_p) { cut = i + 1; break; }
    }
    ids.resize(cut);
    const double inv_t = 1.0 / p.temperature;
    double s2 = 0;
    for (int id : ids) s2 += std::exp((l[(size_t) id] - mx) * inv_t);
    std::vector<std::pair<int, double>> out;
    for (int id : ids) out.push_back({id, std::exp((l[(size_t) id] - mx) * inv_t) / s2});
    return out;
}

// chi-square of `counts` (n draws) against `p`; tokens outside p's support count as an immediate failure
bool chi2_ok(const std::map<int, long>& counts, const std::vector<std::pair<int, double>>& p, long n, const char* what) {
    double x2 = 0;
    int dof = -1;
    long outside = 0;
    std::map<int, double> pm;
    for (auto& [id, pr] : p) pm[id] = pr;
    for (auto& [id, c] : counts)
        if (!pm.count(id)) outside += c;
    double rest_e = 0, rest_o = 0;
    for (auto& [id, pr] : p) {
        const double e = pr * (double) n;
        const double o = counts.count(id) ? (double) counts.at(id) : 0.0;
        if (e < 5) { rest_e += e; rest_o += o; continue; }
        x2 += (o - e) * (o - e) / e;
        ++dof;
    }
    if (rest_e >= 5) { x2 += (rest_o - rest_e) * (rest_o - rest_e) / rest_e; ++dof; }
    // the 99.9% point of chi-square with dof degrees of freedom (Wilson-Hilferty)
    const double z = 3.0902, k = dof > 0 ? dof : 1;
    const double crit = k * std::pow(1.0 - 2.0 / (9.0 * k) + z * std::sqrt(2.0 / (9.0 * k)), 3.0);
    const bool ok = outside == 0 && x2 <= crit;
    std::printf("  %-34s chi2 %8.1f (dof %d, 99.9%% %6.1f), outside p %ld: %s\n", what, x2, dof, crit, outside,
                ok ? "follows p" : "differs from p");
    return ok;
}

}  // namespace

int main() {
    using namespace strata;
    constexpr int NV = 6000, T = 3, TRIALS = 40000;
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.0f, 2.0f);
    std::vector<float> row((size_t) NV);
    for (auto& v : row) v = nd(rng);
    std::vector<float> logits((size_t) T * NV);
    for (int t = 0; t < T; ++t) std::copy(row.begin(), row.end(), logits.begin() + (size_t) t * NV);
    SamplerParams sp;
    sp.top_k = 20;
    sp.top_p = 0.95f;
    sp.temperature = 0.8f;
    sp.seed = 1234;
    const auto P = chain(row, sp);
    // q: p's list with the probabilities reversed (mass moved to the tail) and a token outside p's support
    std::vector<std::pair<int, double>> Q;
    {
        double s = 0;
        for (size_t i = 0; i < P.size(); ++i) { Q.push_back({P[i].first, P[P.size() - 1 - i].second}); s += Q.back().second; }
        int outsider = 0;
        while (std::any_of(P.begin(), P.end(), [&](auto& e) { return e.first == outsider; })) ++outsider;
        Q.push_back({outsider, 0.1});
        s += 0.1;
        for (auto& e : Q) e.second /= s;
    }
    std::vector<int32_t> qrow(core::kSpecQStride, -1);
    for (size_t i = 0; i < Q.size(); ++i) {
        qrow[i] = Q[i].first;
        const float f = (float) Q[i].second;
        std::memcpy(&qrow[core::kSpecQEntries + i], &f, 4);
    }
    for (size_t i = Q.size(); i < (size_t) core::kSpecQEntries; ++i) { const float z = 0; std::memcpy(&qrow[core::kSpecQEntries + i], &z, 4); }

    parity::Device dev;
    float* d_logits = dev.upload(logits);
    int32_t* d_q = dev.upload(qrow);
    int32_t* d_tok = dev.alloc<int32_t>(T);
    int* d_out = dev.alloc<int>(T);
    std::vector<double> qw, pw;
    for (auto& e : Q) qw.push_back(e.second);
    for (auto& e : P) pw.push_back(e.second);
    std::discrete_distribution<int> qd(qw.begin(), qw.end()), pd(pw.begin(), pw.end());
    std::map<int, long> rows[T], drafts, plain;
    long kept = 0;
    std::vector<int32_t> dt(T - 1);
    std::vector<int> out(T);
    for (int n = 0; n < TRIALS; ++n) {
        dt[0] = Q[(size_t) qd(rng)].first;   // drawn from q
        dt[1] = P[(size_t) pd(rng)].first;   // a point-mass draft (a lookup's): any token, here one from p
        ++drafts[dt[0]];
        parity::copy_in(d_tok, dt.data(), dt.size());
        SamplerParams s = sp;
        s.counter = (uint64_t) n * 64;
        if (!kernels::sample_tokens_spec(d_logits, T, NV, nullptr, 0, s, d_tok, d_q, 1, d_out, parity::stream())) {
            std::printf("spec_prob_parity: the split sampler cannot run here\n");
            return 1;
        }
        parity::copy_out(out.data(), d_out, T);
        for (int t = 0; t < T; ++t) ++rows[t][out[(size_t) t]];
        kept += out[0] == dt[0];
        if (n < 4000) {   // control: the plain sampler at the same counters
            kernels::sample_tokens(d_logits, 1, NV, nullptr, 0, s, d_out, parity::stream());
            parity::copy_out(out.data(), d_out, 1);
            ++plain[out[0]];
        }
    }
    double accept = 0;   // the expected acceptance sum min(p, q)
    for (auto& [id, pr] : P)
        for (auto& [qi, qp] : Q)
            if (qi == id) accept += std::min(pr, qp);
    std::printf("spec_prob_parity: %d windows, %zu tokens in p, row 0 kept its q draft %.3f (min(p, q) sums to %.3f)\n",
                TRIALS, P.size(), (double) kept / TRIALS, accept);
    bool ok = chi2_ok(rows[0], P, TRIALS, "row 0 (q draft)");
    ok = chi2_ok(rows[1], P, TRIALS, "row 1 (point-mass draft)") && ok;
    ok = chi2_ok(rows[2], P, TRIALS, "row 2 (no draft)") && ok;
    ok = chi2_ok(plain, P, 4000, "control: sample_tokens") && ok;
    const bool q_fails = !chi2_ok(drafts, P, TRIALS, "control: the drafts (q), must differ");
    ok = ok && q_fails && std::fabs((double) kept / TRIALS - accept) < 0.02;
    std::printf("spec_prob_parity: %s\n", ok ? "ok" : "FAILED");
    return ok ? 0 : 1;
}
