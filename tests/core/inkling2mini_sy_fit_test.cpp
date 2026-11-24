// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// tests/core/inkling2mini_sy_fit_test.cpp - the SYCL runner's memory need per slot, how many slots a card can hold, and
// the sliding-window K/V ring's size (no GPU, no SYCL).
// Upstream's Arc A750 (8 GB), `--parallel 32 --max-context 4096`, UD-IQ1_S: the start died mid-load with "dalloc
// (268435456 bytes): out of device memory" (upstream de1768f2b); a ring of exactly the window gave wrong logits past
// 1024 tokens (upstream #6878).
#include "inkling2mini_sy_fit.h"

#include <cstdio>
#include <string>

using namespace inkling2mini_sycl;

namespace {
int failed = 0;
#define CHECK(expr)                                                                         \
    do {                                                                                    \
        if (!(expr)) {                                                                      \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expr);            \
            ++failed;                                                                       \
        }                                                                                   \
    } while (0)

// Inkling-2-Mini: 24 physical layers, core 6..18 looped 3 times -> 48 visits; global layers 5, 11, 17, 23 (11 and 17
// are in the loop) -> 8 global visits.
Inkling2MiniSyFitGeom inkling2mini(bool dspark, bool vision) {
    Inkling2MiniSyFitGeom g;
    const bool is_global[24] = {0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1};
    for (int v = 0; v < 48; ++v) {
        const int ip = v < 6 ? v : v < 6 + 36 ? 6 + (v - 6) % 12 : 18 + (v - 42);
        g.visit_is_swa.push_back(is_global[ip] ? 0 : 1);
    }
    g.dspark = dspark;
    g.vision = vision;
    return g;
}
constexpr size_t MiB = 1ull << 20u;
constexpr size_t GiB = 1ull << 30u;
}  // namespace

int main() {
    // the ring: the window plus the step's capacity in whole 64-token pages, capped at the context
    CHECK(inkling2mini_sy_batch_cap(1, 512) == 512);
    CHECK(inkling2mini_sy_batch_cap(1, 100) == 256);
    CHECK(inkling2mini_sy_batch_cap(64, 1) == 256);
    CHECK(inkling2mini_sy_swa_ring(1024, 512, 32768) == 1536);
    CHECK(inkling2mini_sy_swa_ring(1024, 300, 32768) == 1344);          // 1324 rounded up to a page
    CHECK(inkling2mini_sy_swa_ring(1024, 4096, 2048) == 2048);          // never past the context
    CHECK(inkling2mini_sy_swa_ring(1024, 512, 256) == 1024);            // never below the window
    CHECK(inkling2mini_sy_swa_ring(1024, 512, 32768, 1024) == 1024);    // INKLING2MINI_SWA_RING=1024: the old ring
    CHECK(inkling2mini_sy_swa_ring(1024, 512, 32768, 2048) == 2048);
    CHECK(inkling2mini_sy_swa_ring(1024, 512, 32768, 100) == 1536);     // below the window: ignored
    // the step margin: what a step of one slot may span; 0 when the ring cannot wrap or is the bare window
    CHECK(inkling2mini_sy_swa_span_cap(1024, 1536, 32768) == 512);
    CHECK(inkling2mini_sy_swa_span_cap(1024, 1024, 32768) == 0);
    CHECK(inkling2mini_sy_swa_span_cap(1024, 2048, 2048) == 0);
    // a step of S + 1 positions in one slot keeps every key its queries read with a ring of window + S: the oldest key
    // the first query reads, (lo - window + 1), is overwritten only by position (lo - window + 1 + ring) > hi
    for (int lo = 0; lo < 4096; lo += 61) {
        const int win = 1024, ring = inkling2mini_sy_swa_ring(win, 512, 32768), hi = lo + 512;
        CHECK((lo - win + 1) + ring > hi);
    }

    const Inkling2MiniSyFitGeom g = inkling2mini(false, false);
    int n_global = 0;
    for (uint8_t s : g.visit_is_swa) n_global += s ? 0 : 1;
    CHECK(g.visit_is_swa.size() == 48 && n_global == 8);

    // the K/V of one slot at 32768 tokens with --chunk 256: 8 global visits of 32768 tokens + 40 window visits of the
    // 1280-position ring (the runner logs it as "KV=1224.0 MiB")
    const size_t one = inkling2mini_sy_runner_reserve(g, 1, 32768, 256);
    const size_t kv_only_1 = 2ull * 1 * (8ull * 32768 + 40ull * 1280) * 8 * 128 * 2;
    CHECK(kv_only_1 == 1224 * MiB);
    CHECK(one > kv_only_1 && one < kv_only_1 + 1 * GiB);

    // 32 slots at 4096 tokens: about 4.6 GiB of K/V alone (the 256 MiB allocation of the crash is one global visit)
    const size_t kv32 = 2ull * 32 * (8ull * 4096 + 40ull * 1280) * 8 * 128 * 2;
    CHECK(32ull * 4096 * 1024 * 2 == 256 * MiB);
    CHECK(inkling2mini_sy_runner_reserve(g, 32, 4096, 256) > kv32);

    // the step buffers grow with the capacity; the logits of 64 rows are in them
    CHECK(inkling2mini_sy_batch_bytes(g, 512) > inkling2mini_sy_batch_bytes(g, 256));
    CHECK(inkling2mini_sy_batch_bytes(g, 256) > 64ull * 201024 * 4);

    // an 8 GB card with about 4 GiB left after the weights: 32 slots at 4096 do not fit, a smaller number does, and
    // the message names it
    const size_t avail = 4 * GiB;
    CHECK(inkling2mini_sy_runner_reserve(g, 32, 4096, 256) > avail);
    const int most = inkling2mini_sy_max_parallel(g, 4096, 256, avail);
    CHECK(most >= 1 && most < 32);
    CHECK(inkling2mini_sy_runner_reserve(g, most, 4096, 256) <= avail);
    CHECK(inkling2mini_sy_runner_reserve(g, most + 1, 4096, 256) > avail);
    const std::string msg = inkling2mini_sy_fit_message(g, 32, 4096, 256, avail);
    CHECK(!msg.empty());
    CHECK(msg.find("--parallel 32") != std::string::npos);
    CHECK(msg.find("--parallel " + std::to_string(most)) != std::string::npos);

    // what fits gives no message; the reserve grows with slots, with the context and with the chunk (the ring)
    CHECK(inkling2mini_sy_fit_message(g, 1, 4096, 256, avail).empty());
    CHECK(inkling2mini_sy_fit_message(g, most, 4096, 256, avail).empty());
    CHECK(inkling2mini_sy_runner_reserve(g, 2, 4096, 256) > inkling2mini_sy_runner_reserve(g, 1, 4096, 256));
    CHECK(inkling2mini_sy_runner_reserve(g, 1, 8192, 256) > inkling2mini_sy_runner_reserve(g, 1, 4096, 256));
    CHECK(inkling2mini_sy_runner_reserve(g, 1, 8192, 1024) > inkling2mini_sy_runner_reserve(g, 1, 8192, 256));

    // not even one slot: the message says so and asks for a shorter context
    const std::string none = inkling2mini_sy_fit_message(g, 1, 1000000, 256, avail);
    CHECK(!none.empty() && none.find("shorter --max-context") != std::string::npos);
    CHECK(inkling2mini_sy_max_parallel(g, 1000000, 256, avail) == 0);

    // DSpark and Vision add to the need
    CHECK(inkling2mini_sy_runner_reserve(inkling2mini(true, false), 4, 8192, 256) >
          inkling2mini_sy_runner_reserve(g, 4, 8192, 256));
    CHECK(inkling2mini_sy_runner_reserve(inkling2mini(false, true), 4, 8192, 256) ==
          inkling2mini_sy_runner_reserve(g, 4, 8192, 256) + 700 * MiB);

    if (failed != 0) {
        std::fprintf(stderr, "inkling2mini_sy_fit_test: %d check(s) failed\n", failed);
        return 1;
    }
    std::printf("inkling2mini_sy_fit_test: OK\n");
    return 0;
}
