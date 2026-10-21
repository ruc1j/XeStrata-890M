// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/kernels/ecap_parity.cpp - the verify window's expert-cap redirect (STRATA_VERIFY_ECAP, upstream PR 3545) against
// its contract.  `verify_ecap` rewrites the window's routing in place before resident_plan: token 0 keeps all k picks,
// the speculative tail keeps the top (cap - k) entries by (weight desc, index asc), and every dropped entry is
// redirected round-robin onto a kept expert with weight 0.  Checked on random data shaped like the router (512
// experts, top-10, windows of 2..6 tokens):
//   * cap >= the entry count (or n <= k, or cap <= k): a no-op - ids and weights bitwise unchanged
//   * token 0's entries are never touched
//   * every kept entry keeps its id and weight; every dropped entry's weight is exactly 0 and its id is one of the
//     kept set's (so the plan never meets an expert the uncapped window would not have named)
//   * the same input gives the same output (the fold is deterministic)
// GPU, synthetic, no model.
#include "strata/kernels/verify_kernels.hpp"

#include "parity_device.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace k = strata::kernels;
namespace P = strata::parity;

int main() {
    const int E = 512, K = 10;
    int bad = 0;
    std::mt19937 rng(3545);  // NOLINT(bugprone-random-generator-seed): reproducible fixtures.
    std::uniform_real_distribution<float> w_dist(0.0f, 1.0f);
    void* st = P::stream();
    auto* d_ids = static_cast<int32_t*>(P::alloc_bytes(128 * 4));
    auto* d_w = static_cast<float*>(P::alloc_bytes(128 * 4));
    for (int T : {2, 3, 4, 6}) {
        const int n = T * K;
        std::vector<int32_t> ids(n), ids0;
        std::vector<float> w(n), w0;
        for (int t = 0; t < T; ++t) {   // distinct ids per token, ascending, as the router's top-k produces
            std::vector<int> pool(E);
            for (int i = 0; i < E; ++i) pool[i] = i;
            std::shuffle(pool.begin(), pool.end(), rng);
            std::sort(pool.begin(), pool.begin() + K);
            for (int j = 0; j < K; ++j) ids[t * K + j] = pool[j];
        }
        for (auto& v : w) v = w_dist(rng);
        ids0 = ids; w0 = w;
        auto up = [&] { P::copy_bytes_in(d_ids, ids0.data(), (size_t) n * 4); P::copy_bytes_in(d_w, w0.data(), (size_t) n * 4); };
        auto down = [&] { P::sync(); P::copy_bytes_out(ids.data(), d_ids, (size_t) n * 4); P::copy_bytes_out(w.data(), d_w, (size_t) n * 4); };
        up();
        k::verify_ecap(d_ids, d_w, n, K, n + 7, st);   // a cap that keeps everything: bitwise a no-op
        down();
        const bool noop = std::memcmp(ids.data(), ids0.data(), (size_t) n * 4) == 0 &&
                          std::memcmp(w.data(), w0.data(), (size_t) n * 4) == 0;
        std::printf("  cap >= n     T %d  %s\n", T, noop ? "no-op" : "*** CHANGED ***");
        if (!noop) ++bad;
        for (int cap : {K + 1, 16, 20, 26}) {
            if (cap >= n) continue;
            // the expected kept set: token 0's k, then the top (cap - k) tail entries by (w desc, index asc)
            std::vector<int> tail(n - K);
            for (int j = K; j < n; ++j) tail[j - K] = j;
            std::stable_sort(tail.begin(), tail.end(), [&](int a, int b) { return w0[a] > w0[b]; });
            std::vector<char> kept(n, 0);
            for (int j = 0; j < K; ++j) kept[j] = 1;
            for (int i = 0; i < cap - K && i < (int) tail.size(); ++i) kept[tail[i]] = 1;
            up();
            k::verify_ecap(d_ids, d_w, n, K, cap, st);
            down();
            const std::vector<int32_t> ids_a = ids;
            const std::vector<float> w_a = w;
            up();   // the same input again: the fold is deterministic
            k::verify_ecap(d_ids, d_w, n, K, cap, st);
            down();
            bool ok = ids_a == ids && w_a == w;
            std::vector<int32_t> kept_ids;
            for (int j = 0; j < n; ++j) if (kept[j]) kept_ids.push_back(ids0[j]);
            for (int j = 0; j < n && ok; ++j) {
                if (kept[j]) ok = ids[j] == ids0[j] && w[j] == w0[j];   // token 0 and the kept tail: untouched
                else ok = w[j] == 0.0f && std::find(kept_ids.begin(), kept_ids.end(), ids[j]) != kept_ids.end();
            }
            int dropped = 0;
            for (int j = K; j < n; ++j) if (!kept[j]) ++dropped;
            std::printf("  cap %2d       T %d  kept %2d dropped %2d  %s\n", cap, T, (int) kept_ids.size(), dropped,
                        ok ? "ok" : "*** MISMATCH ***");
            if (!ok) ++bad;
        }
    }
    std::printf("ecap_parity: %s\n", bad == 0 ? "PASS" : "FAIL");
    return bad == 0 ? 0 : 1;
}
