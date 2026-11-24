// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// inkling2mini_sy_fit.h - what the SYCL runner needs per slot and in total, how many slots fit a card, and the size of
// the sliding-window K/V ring (no SYCL, unit-tested by tests/core/inkling2mini_sy_fit_test.cpp)
//
// The SYCL runner allocates every slot's K/V (fp16: the whole context for a global visit, the ring for a window visit),
// the ShortConv state, the DSpark context and its batch buffers at start.  With `--parallel 32 --max-context 4096` that
// is several GiB of K/V alone: upstream's 8 GB Arc A750 died in the middle of the load with "dalloc ... out of device
// memory".  The same numbers drive the expert planner's reserve, so they live here; the backend checks them against
// the card's free memory BEFORE the runner allocates and refuses with a plain message that names the largest
// --parallel that fits (upstream de1768f2b).
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace inkling2mini_sycl {

// The tokens one runner step holds: the backend's batch capacity, as the runner sizes its buffers.
inline int inkling2mini_sy_batch_cap(int par, int chunk) { return std::max(256, std::max(chunk, par)); }

// The positions of a slot's sliding-window K/V ring.  A step stores the K/V of all its tokens before attention reads
// the ring, so a ring of exactly swa_win positions lets a later token of the same step overwrite a key that an earlier
// query of the step still needs: (p - swa_win + 1) and (p + 1) share a ring slot (upstream #6878: wrong logits for
// prompts over 1024 tokens).  A ring of swa_win + S positions is safe for any step whose positions in one slot span at
// most S + 1, so the ring gets the step's capacity as margin, rounded up to whole 64-token pages and capped at the
// context (positions below the ring size never wrap).  `ring_override` > 0 (INKLING2MINI_SWA_RING) replaces the
// margin; swa_win itself brings back the old ring and its overwrite, for A/B checks only.
inline int inkling2mini_sy_swa_ring(int swa_win, int batch_cap, int max_ctx, int ring_override = 0) {
    const int win = std::max(1, swa_win);
    int ring = win + std::max(batch_cap, 64);
    if (ring_override >= win) ring = ring_override;
    ring = (ring + 63) / 64 * 64;
    const int ctx_cap = (std::max(max_ctx, win) + 63) / 64 * 64;
    return std::max(win, std::min(ring, ctx_cap));
}

// INKLING2MINI_SWA_RING=N, or 0 when unset.
inline int inkling2mini_sy_swa_ring_env() {
    const char* e = std::getenv("INKLING2MINI_SWA_RING");
    return (e != nullptr && *e != 0) ? static_cast<int>(std::strtol(e, nullptr, 10)) : 0;
}

// The most tokens of one slot a step may span so the ring keeps every key its queries read: ring - swa_win when the
// ring can wrap (it is shorter than the context), else 0 (no limit).
inline int inkling2mini_sy_swa_span_cap(int swa_win, int ring, int max_ctx) {
    const int win = std::max(1, swa_win);
    const int ctx_cap = (std::max(max_ctx, win) + 63) / 64 * 64;
    return (ring > win && ring < ctx_cap) ? (ring - win) : 0;
}

struct Inkling2MiniSyFitGeom {
    std::vector<uint8_t> visit_is_swa;  // [n_visits], 1 = sliding window layer, 0 = global
    int swa_win = 1024;
    int n_kv = 8;
    int head_dim = 128;
    int hidden = 2304;
    int gate_rank = 64;
    int vocab_padded = 201024;
    int swa_ring_override = 0;  // INKLING2MINI_SWA_RING
    bool dspark = false;
    bool vision = false;
};

// The bytes of the runner's step buffers for a capacity of `mb` tokens: the dalloc list of Inkling2MiniSyRunner::init
// as a formula (upstream 1999e3e82 counts its HIP backend's batch scratch the same way), plus the logits of up to 64
// rows and the verify ring of the ShortConv state of every visit.
inline size_t inkling2mini_sy_batch_bytes(const Inkling2MiniSyFitGeom& g, int mb) {
    const size_t b = (size_t)std::max(64, mb);
    const size_t D = (size_t)g.hidden;
    // [MB, D] fp32: image embeddings, cur, res, 6 blocks, prelude, 2 bank blocks, bank out, 4 messages, 3 inputs,
    // 3 controller concats of 2 D, wo raw / out, MoE input, 8 expert outputs, MoE raw / out, head out / norm, 4 taps
    size_t per_tok = 45 * D * 4;
    // q, k / v raw and conv, r, conv taps, sinks, gate, attention out, router logits, top-k weights, expert hidden,
    // controller gates (fp32); ids, positions, top-k ids and pairs, argmax (int32); image flags, int8 activations
    per_tok += (3072ull * 3 + 1024ull * 4 + 384 + 2 * 32 + 24 + 128 + 8 + 8 * 1152 + 3ull * (size_t)g.gate_rank + 1)
               * 4;
    per_tok += (4 + 8 + 8 + 1) * 4 + 1 + 9216;
    if (g.dspark) per_tok += D * 4;                                    // f_rnn of each token
    size_t t = b * per_tok;
    t += (size_t)std::min<size_t>(b, 64) * (size_t)g.vocab_padded * 4;  // logits
    t += g.visit_is_swa.size() * 8ull * 3 * (2ull * 1024 + 2 * D) * 4;  // ShortConv verify ring
    return t;
}

// The bytes the runner, its batch buffers, DSpark, Vision, the hybrid cache ring and the XMX scratch need on the device
// for `par` slots at `ctx` tokens (the planner keeps this much free of experts).
inline size_t inkling2mini_sy_runner_reserve(const Inkling2MiniSyFitGeom& g, int par, int ctx, int chunk) {
    const size_t p = (size_t)std::max(1, par);
    const int batch_cap = inkling2mini_sy_batch_cap(par, chunk);
    const size_t ring = (size_t)inkling2mini_sy_swa_ring(g.swa_win, batch_cap, ctx, g.swa_ring_override);
    size_t kv = 0;
    for (uint8_t swa : g.visit_is_swa) {
        const size_t stride = swa ? ring : (size_t)ctx;
        kv += 2ull * p * stride * (size_t)g.n_kv * (size_t)g.head_dim * 2ull;
    }
    const size_t dsp = g.dspark ? (3ull * 2ull * p * (size_t)ctx * 1024ull * 2ull) : 0ull;
    const size_t sc = g.visit_is_swa.size() * p * 3ull * (2048ull + 2ull * (size_t)g.hidden) * 8ull;
    const size_t act = inkling2mini_sy_batch_bytes(g, batch_cap) + (64ull << 20);
    const size_t vis = g.vision ? (700ull << 20) : 0ull;
    const size_t hyb_xmx = 320ull << 20;  // elastic VRAM cache ring (<= 256 MiB) + XMX scratch + ping-pong
    return kv + dsp + sc + act + vis + hyb_xmx;
}

// The largest --parallel (1..64) whose reserve fits `avail` bytes; 0 when even one slot does not.
inline int inkling2mini_sy_max_parallel(const Inkling2MiniSyFitGeom& g, int ctx, int chunk, size_t avail) {
    int best = 0;
    for (int p = 1; p <= 64; ++p) {
        if (inkling2mini_sy_runner_reserve(g, p, ctx, chunk) <= avail) best = p;
        else break;
    }
    return best;
}

// The refusal text; empty when the request fits `avail`.
inline std::string inkling2mini_sy_fit_message(const Inkling2MiniSyFitGeom& g, int par, int ctx, int chunk,
                                               size_t avail) {
    const size_t need = inkling2mini_sy_runner_reserve(g, par, ctx, chunk);
    if (need <= avail) return std::string();
    const double gib = 1024.0 * 1024.0 * 1024.0;
    char buf[480];
    const int most = inkling2mini_sy_max_parallel(g, ctx, chunk, avail);
    if (most >= 1) {
        std::snprintf(buf, sizeof buf,
                      "sycl: --parallel %d at --max-context %d needs about %.1f GiB of this GPU for the K/V caches "
                      "and buffers, but only %.1f GiB are free after the weights; "
                      "the most that fits is --parallel %d (or a shorter --max-context)",
                      par, ctx, need / gib, avail / gib, most);
    } else {
        std::snprintf(buf, sizeof buf,
                      "sycl: --max-context %d needs about %.1f GiB of this GPU for the K/V caches and buffers even "
                      "for one slot, but only %.1f GiB are free after the weights; use a shorter --max-context",
                      ctx, need / gib, avail / gib);
    }
    return std::string(buf);
}

}  // namespace inkling2mini_sycl
