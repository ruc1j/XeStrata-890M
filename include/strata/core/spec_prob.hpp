// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// include/strata/core/spec_prob.hpp - PROBABILISTIC DRAFT ACCEPTANCE (STRATA_SPEC_PROB=1, off by default; upstream
// 33356f72, 589ccc47).
//
// Speculative rejection sampling (Leviathan et al. / Chen et al.) for the MTP drafts of a SAMPLED request.  Under
// sampling the verify window accepts a draft only when it EQUALS the token the target sampled for that row (exact
// match).  Here the draft layer SAMPLES its draft d from its own distribution q (its own Philox stream) and the
// verifier accepts d with probability min(1, p(d) / q(d)); on a reject it samples the replacement from the residual
// norm(max(0, p - q)).  The output token of every row is then distributed exactly as p, the target's chain (penalties
// -> top_k -> top_p -> min_p -> temperature), whatever q is.
//
//   p   the TARGET'S distribution at the row: the list the sampler really samples from, after penalties, top_k,
//       top_p, min_p and temperature, renormalised.  At most 64 entries.  NOT the raw softmax: using it would accept
//       tokens the chain never emits (the llama.cpp #29924 class of bug).
//   q   the DRAFTER'S distribution: the same chain over the draft head's logits.  A sparse list of at most 64
//       (id, probability) pairs, probabilities summing to 1.
//
// "NO q MEANS POINT MASS, NEVER q = 0."  A drafter without a candidate list (prompt lookup, suffix drafts, the lookup
// chain) proposes one token for sure: q is a delta at d, the accept probability is p(d) and the residual is p with d
// removed.  The engine only builds q lists for the MTP drafts; every other row of a window is a point-mass row.
//
// --spec-min-p gates a further draft on q's TOP probability (how sure the draft head is), never on the drawn token's
// own: a gate on the drawn token would make the proposal depend on the draw, and the output would no longer follow p
// (upstream PR #3416's example: p = (0.5, 0.5), q = (0.875, 0.125), threshold 0.5 gives (0.5625, 0.4375)).
//
// THE VERIFY ROWS ARE INDEPENDENT.  Row t of a window at pos0 judges draft t (token t + 1 of the window) with its own
// uniforms, disjoint Philox counter spaces keyed by the row's position pos0 + t:
//     u_pick   counter = pos          the plain sample (the window's last row: the bonus token) - the same draw the
//                                     exact-match sampler uses
//     u_accept counter = pos | 1<<62  accept / reject
//     u_resid  counter = pos | 1<<63  the residual sample after a reject
//     (the drafter's own draw: counter = pos | 1<<61)
// Row t writes out[t] = d when it accepts, and the residual sample (never d) when it rejects, so the host's unchanged
// loop `while (a < T-1 && window[a+1] == out[a]) ++a` takes exactly the accepted prefix, and out[a] is the bonus token.
#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace strata::core {

/// STRATA_SPEC_PROB=1: the mode (it runs on the coupled drafter's machinery, coupled_draft.hpp).
inline bool spec_prob_env() {
    static const bool on = [] {
        const char* e = std::getenv("STRATA_SPEC_PROB");
        return e != nullptr && *e != '\0' && std::strcmp(e, "0") != 0;
    }();
    return on;
}

/// STRATA_SPEC_MIN_TEMP=x: sampled drafting only for requests with temperature >= x (default 0: every sampled
/// request); below x the drafter keeps its argmax drafts and the exact-match rule.  Upstream measured it gaining at
/// temperature 1.0 and losing below about 0.7 on the RTX 3060 and 5070.
inline float spec_min_temp() {
    static const float s = [] {
        const char* e = std::getenv("STRATA_SPEC_MIN_TEMP");
        return e != nullptr ? (float) std::atof(e) : 0.0f;
    }();
    return s;
}

/// The mode as the engine runs it: STRATA_SPEC_PROB, or its default on the device (mtp.cpp); and the lowest
/// temperature it takes (STRATA_SPEC_MIN_TEMP, or the default's).
bool spec_prob_on();
float spec_prob_min_temp();
/// XeStrata-890M: a sampled request's window under the mode at most this many tokens (STRATA_SPEC_PROB_MAX_T; the
/// default's 4, 0 elsewhere: --spec's).  On the 890M the sampled drafts pay up to 4 and greedy ones up to 6 (mtp.cpp).
int spec_prob_max_t();
/// XeStrata-890M: the drafter's q over the draft head's one-bit sketch candidates, without the penalties (any q keeps
/// the target's distribution); STRATA_SPEC_PROB_SKETCH=0: over the whole head with the penalties, as upstream.
bool spec_prob_sketch();

/// One q row = 128 int32 slots: 64 token ids (-1 ends the list), then the 64 probabilities as float bits.
inline constexpr int kSpecQEntries = 64;
inline constexpr int kSpecQStride = 2 * kSpecQEntries;

inline constexpr uint64_t kSpecCtrAccept = 1ull << 62;
inline constexpr uint64_t kSpecCtrResid = 1ull << 63;
inline constexpr uint64_t kSpecCtrDraft = 1ull << 61;

}  // namespace strata::core
