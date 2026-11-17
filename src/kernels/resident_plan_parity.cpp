// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/kernels/resident_plan_parity.cpp - the verify window's device-built expert plan (#783 PR-e: the groups numbered
// by a scan) against a host replay of its contract: distinct experts in routing order, each with its entries
// ascending, `ptr[g]` = the expert's slot, counts / start / dst / tok / start2 in the host pool's layout.
//
// n = 1..80 entries (a window of up to kVerifyMaxT tokens x 10 experts), ids drawn from pools of 3..512 experts so
// duplicates run from "every entry the same expert" to "all distinct", with and without a slot table (`slot_off`),
// and the skip word (ring when the plan was built; 0 and the plan untouched when an expert was not resident).
// Every word of the plan the host pool would read is compared with memcmp. GPU, synthetic, no model.
//
// XeStrata-890M: resident_plan_top10 (the router's top-10 and the plan in one launch) against the two launches it
// replaces, native_router_top10_multi and resident_plan: the ids, the weights, the whole plan and the skip word, for
// windows of 1..8 tokens of 512 and of 256 experts, the tokens' logits near each other (shared experts) or apart.
#include "strata/kernels/verify_kernels.hpp"
#include "strata/kernels/native_router.hpp"

#include "parity_device.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace k = strata::kernels;
namespace P = strata::parity;

namespace {
template <typename T> T* dalloc(size_t n) { return static_cast<T*>(P::alloc_bytes(n * sizeof(T) + 64)); }
}  // namespace

int main() {
    const int K = 10, NE = 512;
    const long long capx = (long long) k::kVerifyMaxT * K;
    const long long ptr_off = ((4 + (capx + 1) + 2 * capx) + 1) & ~1ll;
    const size_t plan_words = (size_t) (ptr_off + 4 * capx + capx + 16);
    std::mt19937 rng(783);  // NOLINT(bugprone-random-generator-seed): reproducible fixtures.
    const long long blob = 1337;
    uint8_t* const cache_base = reinterpret_cast<uint8_t*>(uintptr_t(0x40000000));   // never dereferenced
    void* st = P::stream();

    auto* d_ids = dalloc<int32_t>(128);
    auto* d_res = dalloc<int32_t>(NE);
    auto* d_off = dalloc<unsigned long long>(1024);
    auto* d_plan = dalloc<int32_t>(plan_words);
    auto* d_skip = dalloc<uint32_t>(1);

    std::vector<int32_t> res(NE);
    for (int e = 0; e < NE; ++e) res[e] = (e * 37 + 11) % 1000;      // distinct slots, not the identity
    std::vector<unsigned long long> off(1024);
    for (int s = 0; s < 1024; ++s) off[s] = (unsigned long long) s * 4096 + (unsigned long long) (s % 7) * 64;
    P::copy_bytes_in(d_res, res.data(), (size_t) NE * 4);
    P::copy_bytes_in(d_off, off.data(), off.size() * 8);
    const std::vector<int32_t> fill(plan_words, (int32_t) 0xABABABABu);
    const uint32_t zero = 0;

    int cases = 0, bad = 0;
    for (int n = 1; n <= 80; ++n) {
        for (int variant = 0; variant < 4; ++variant) {
            const int pool = variant == 0 ? 3 : (variant == 1 ? 12 : (variant == 2 ? 40 : 512));
            std::vector<int32_t> ids(n);
            for (int i = 0; i < n; ++i) ids[i] = (int32_t) ((rng() % pool) * (512 / pool));
            const bool use_off = (n + variant) % 2 == 1;
            P::copy_bytes_in(d_ids, ids.data(), (size_t) n * 4);

            // host replay
            std::vector<int32_t> counts = {0, n, 0}, start, dst(n), tok(n);
            std::vector<unsigned long long> ptr;
            std::vector<int> first_seen;     // group -> expert
            for (int i = 0; i < n; ++i) {
                bool seen = false;
                for (int e : first_seen) seen = seen || e == ids[i];
                if (!seen) first_seen.push_back(ids[i]);
            }
            const int groups = (int) first_seen.size();
            counts[0] = groups;
            start.assign(groups + 1, 0);
            int pos = 0;
            for (int g = 0; g < groups; ++g) {
                start[g] = pos;
                ptr.push_back((unsigned long long) (uintptr_t) cache_base +
                              (use_off ? off[res[first_seen[g]]] : (unsigned long long) res[first_seen[g]] * (unsigned long long) blob));
                for (int i = 0; i < n; ++i)
                    if (ids[i] == first_seen[g]) { dst[pos] = i; tok[pos] = i / K; ++pos; }
            }
            start[groups] = n;

            P::copy_bytes_in(d_plan, fill.data(), plan_words * 4);
            P::copy_bytes_in(d_skip, &zero, 4);
            k::resident_plan(d_ids, n, K, d_res, NE, cache_base, use_off ? d_off : nullptr, blob, d_plan, capx, d_skip,
                             7u, st);
            P::sync();
            std::vector<int32_t> plan(plan_words);
            P::copy_bytes_out(plan.data(), d_plan, plan_words * 4);
            uint32_t skip = 0;
            P::copy_bytes_out(&skip, d_skip, 4);
            bool ok = skip == 7u;
            ok = ok && std::memcmp(plan.data(), counts.data(), counts.size() * 4) == 0;
            ok = ok && std::memcmp(plan.data() + 4, start.data(), (size_t) (groups + 1) * 4) == 0;
            const int32_t* pdst = plan.data() + 4 + capx + 1;
            ok = ok && std::memcmp(pdst, dst.data(), (size_t) n * 4) == 0;
            ok = ok && std::memcmp(pdst + capx, tok.data(), (size_t) n * 4) == 0;
            ok = ok && std::memcmp(plan.data() + ptr_off, ptr.data(), (size_t) groups * 8) == 0;
            ok = ok && plan[(size_t) (ptr_off + 4 * capx)] == n;
            ++cases;
            if (!ok) {
                std::printf("    *** n=%d pool=%d %s: the plan differs from the host replay ***\n", n, pool,
                            use_off ? "slot_off" : "blob");
                ++bad;
            }
        }
    }

    // a routed expert that is not resident: the kernel leaves the plan alone and zeroes *skip
    {
        std::vector<int32_t> res2 = res;
        res2[5] = -1;
        P::copy_bytes_in(d_res, res2.data(), (size_t) NE * 4);
        const int32_t ids[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        P::copy_bytes_in(d_ids, ids, sizeof ids);
        P::copy_bytes_in(d_plan, fill.data(), plan_words * 4);
        const uint32_t seven = 7;
        P::copy_bytes_in(d_skip, &seven, 4);
        k::resident_plan(d_ids, 10, K, d_res, NE, cache_base, nullptr, blob, d_plan, capx, d_skip, 7u, st);
        P::sync();
        std::vector<int32_t> plan(plan_words);
        P::copy_bytes_out(plan.data(), d_plan, plan_words * 4);
        uint32_t skip = 99;
        P::copy_bytes_out(&skip, d_skip, 4);
        bool untouched = true;
        for (int32_t w : plan) untouched = untouched && (uint32_t) w == 0xABABABABu;
        ++cases;
        if (skip != 0 || !untouched) {
            std::printf("    *** a non-resident expert: skip=%u untouched=%d (want 0 / 1) ***\n", skip, (int) untouched);
            ++bad;
        }
    }

    // the top-10 and the plan in one launch: the two launches' bits
    P::copy_bytes_in(d_res, res.data(), (size_t) NE * 4);
    {
        auto* d_logits = dalloc<float>((size_t) 8 * NE);
        auto* d_w = dalloc<float>(128);
        std::normal_distribution<float> g(0.0f, 1.0f);
        for (const int ne : {512, 256}) {
            for (int n_tok = 1; n_tok <= 8; ++n_tok) {
                for (int variant = 0; variant < 6; ++variant) {
                    // variant 0..2: the tokens share most of their experts (noise on one row); 3..5: independent rows
                    std::vector<float> logits((size_t) n_tok * ne), row((size_t) ne);
                    for (auto& v : row) v = 3.0f * g(rng);
                    for (int t = 0; t < n_tok; ++t)
                        for (int e = 0; e < ne; ++e)
                            logits[(size_t) t * ne + e] = variant < 3 ? row[(size_t) e] + 0.3f * (float) variant * g(rng)
                                                                      : 3.0f * g(rng);
                    const int n = n_tok * K;
                    const bool use_off = (n_tok + variant) % 2 == 1;
                    P::copy_bytes_in(d_logits, logits.data(), logits.size() * 4);
                    std::vector<int32_t> ids_a(n), ids_b(n), plan_a(plan_words), plan_b(plan_words);
                    std::vector<float> w_a(n), w_b(n);
                    uint32_t skip_a = 0, skip_b = 0;
                    // the two launches
                    P::copy_bytes_in(d_plan, fill.data(), plan_words * 4);
                    P::copy_bytes_in(d_skip, &zero, 4);
                    k::native_router_top10_multi(d_logits, d_ids, d_w, n_tok, st, ne);
                    k::resident_plan(d_ids, n, K, d_res, ne, cache_base, use_off ? d_off : nullptr, blob, d_plan, capx,
                                     d_skip, 9u, st, 0, d_w);
                    P::sync();
                    P::copy_bytes_out(ids_a.data(), d_ids, (size_t) n * 4);
                    P::copy_bytes_out(w_a.data(), d_w, (size_t) n * 4);
                    P::copy_bytes_out(plan_a.data(), d_plan, plan_words * 4);
                    P::copy_bytes_out(&skip_a, d_skip, 4);
                    // the one launch, over buffers that hold other values
                    const std::vector<int32_t> junk(128, -7);
                    P::copy_bytes_in(d_ids, junk.data(), (size_t) 128 * 4);
                    P::copy_bytes_in(d_w, junk.data(), (size_t) 128 * 4);
                    P::copy_bytes_in(d_plan, fill.data(), plan_words * 4);
                    P::copy_bytes_in(d_skip, &zero, 4);
                    k::resident_plan_top10(d_logits, d_ids, d_w, n_tok, d_res, ne, cache_base, use_off ? d_off : nullptr,
                                           blob, d_plan, capx, d_skip, 9u, st);
                    P::sync();
                    P::copy_bytes_out(ids_b.data(), d_ids, (size_t) n * 4);
                    P::copy_bytes_out(w_b.data(), d_w, (size_t) n * 4);
                    P::copy_bytes_out(plan_b.data(), d_plan, plan_words * 4);
                    P::copy_bytes_out(&skip_b, d_skip, 4);
                    const bool ok = skip_a == 9u && skip_b == 9u && ids_a == ids_b &&
                                    std::memcmp(w_a.data(), w_b.data(), (size_t) n * 4) == 0 && plan_a == plan_b;
                    ++cases;
                    if (!ok) {
                        std::printf("    *** %d experts, %d tokens, variant %d: the one launch differs from the two "
                                    "(skip %u / %u) ***\n", ne, n_tok, variant, skip_a, skip_b);
                        ++bad;
                    }
                }
            }
        }
        // an expert of the top-10 not resident: *skip = 0, the plan untouched, the ids and weights still the router's
        std::vector<int32_t> none((size_t) NE, -1);
        P::copy_bytes_in(d_res, none.data(), (size_t) NE * 4);
        std::vector<float> logits((size_t) 3 * NE);
        for (auto& v : logits) v = 3.0f * g(rng);
        P::copy_bytes_in(d_logits, logits.data(), logits.size() * 4);
        std::vector<int32_t> ids_a(30), ids_b(30), plan(plan_words);
        k::native_router_top10_multi(d_logits, d_ids, d_w, 3, st, NE);
        P::sync();
        P::copy_bytes_out(ids_a.data(), d_ids, (size_t) 30 * 4);
        P::copy_bytes_in(d_plan, fill.data(), plan_words * 4);
        const uint32_t nine = 9;
        P::copy_bytes_in(d_skip, &nine, 4);
        k::resident_plan_top10(d_logits, d_ids, d_w, 3, d_res, NE, cache_base, nullptr, blob, d_plan, capx, d_skip, 9u,
                               st);
        P::sync();
        P::copy_bytes_out(ids_b.data(), d_ids, (size_t) 30 * 4);
        P::copy_bytes_out(plan.data(), d_plan, plan_words * 4);
        uint32_t skip = 99;
        P::copy_bytes_out(&skip, d_skip, 4);
        bool untouched = true;
        for (int32_t w : plan) untouched = untouched && (uint32_t) w == 0xABABABABu;
        ++cases;
        if (skip != 0 || !untouched || ids_a != ids_b) {
            std::printf("    *** the one launch, a non-resident expert: skip=%u untouched=%d ids %s ***\n", skip,
                        (int) untouched, ids_a == ids_b ? "same" : "differ");
            ++bad;
        }
    }

    std::printf("resident_plan: %d cases, %d failures\n", cases, bad);
    if (bad) return 1;
    std::printf("resident_plan_parity OK\n");
    return 0;
}
