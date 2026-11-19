// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
#include "strata/core/conversation_memory.hpp"

#include <cstdio>
#include <cstdlib>
#include <limits>
#include <optional>
#include <sstream>
#include <vector>

using namespace strata::core;
namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}

// #5697: evict_for_ram over a fake machine.  Four parked conversations of `entry` bytes; an eviction gives back `ratio` of
// its bytes to the reading (0: the C library keeps it).  The park needs `need` bytes above `floor`.
struct FakeRam {
    uint64_t avail = 0;
    std::vector<size_t> parked;
    double ratio = 0.0;
    std::optional<uint64_t> last;
    bool admit(uint64_t need, uint64_t floor) { last = avail; return conversation_memory_admit(last, need, floor); }
    size_t evict() {
        if (parked.empty()) return 0;
        const size_t b = parked.front();
        parked.erase(parked.begin());
        avail += (uint64_t) ((double) b * ratio);
        return b;
    }
};
void evict_for_ram_cases() {
    constexpr uint64_t MiB = uint64_t(1) << 20U, GiB = MiB << 10U;
    const uint64_t floor = 2 * GiB, need = 3 * GiB;
    auto run = [&](FakeRam& m, size_t max) {
        return evict_for_ram([&] { return m.admit(need, floor); }, [&] { return m.last; }, [&] { return m.evict(); }, max);
    };
    {   // the memory comes back: evict until it fits, no further
        FakeRam m{4 * GiB, {GiB, GiB, GiB, GiB}, 1.0, {}};
        const EvictForRam r = run(m, 4);
        check(r.fits && r.evicted == 1 && !r.no_progress && m.parked.size() == 3, "memory that comes back: one eviction, it fits");
    }
    {   // #5697: the memory stays in the heap: one eviction shows nothing, the gate stops, three stay parked
        FakeRam m{4 * GiB, {GiB, GiB, GiB, GiB}, 0.0, {}};
        const EvictForRam r = run(m, 4);
        check(!r.fits && r.evicted == 1 && r.no_progress && m.parked.size() == 3,
              "memory that does not come back: one eviction, then the gate stops; three stay parked");
        // the loop it replaces (generate.cpp before #5697's fix) went on to the slot bound and emptied the cache
        FakeRam old{4 * GiB, {GiB, GiB, GiB, GiB}, 0.0, {}};
        bool fits = old.admit(need, floor);
        size_t evicted = 0;
        while (!fits && evicted < 4 && old.evict() > 0) { ++evicted; fits = old.admit(need, floor); }
        check(!fits && evicted == 4 && old.parked.empty(), "(the old loop: four evictions, an empty cache, refused anyway)");
    }
    {   // half comes back: progress, it goes on until it fits
        FakeRam m{4 * GiB, {GiB, GiB, GiB, GiB}, 0.6, {}};
        const EvictForRam r = run(m, 4);
        check(r.fits && r.evicted == 2 && m.parked.size() == 2, "most of it comes back: two evictions, it fits");
    }
    {   // already fits: nothing is evicted
        FakeRam m{8 * GiB, {GiB, GiB}, 0.0, {}};
        const EvictForRam r = run(m, 4);
        check(r.fits && r.evicted == 0 && m.parked.size() == 2, "a park that fits evicts nothing");
    }
    {   // small entries: a gain of 64 MiB or more counts as progress even under half (the reading moves in coarse steps)
        FakeRam m{4 * GiB + 700 * MiB, {200 * MiB, 200 * MiB, 200 * MiB, 200 * MiB}, 0.4, {}};
        const EvictForRam r = run(m, 4);
        check(r.evicted == 4 && !r.no_progress && !r.fits, "80 MiB back per 200 MiB entry is progress: all four go, still short");
    }
    {   // the slot bound and an empty cache end the loop as before
        FakeRam m{3 * GiB, {GiB, GiB}, 1.0, {}};
        const EvictForRam r = run(m, 1);
        check(!r.fits && r.evicted == 1 && m.parked.size() == 1, "max_evictions bounds the loop");
        FakeRam e{4 * GiB, {}, 1.0, {}};
        check(!run(e, 4).fits && e.parked.empty(), "nothing parked: refused, nothing to evict");
    }
    {   // #2253: short by more than every parked conversation holds: nothing is evicted, the four stay parked
        auto run2253 = [&](FakeRam& m) {
            return evict_for_ram([&] { return m.admit(need, floor); }, [&] { return m.last; }, [&] { return m.evict(); }, 4,
                                 need + floor, (uint64_t) 4 * GiB);
        };
        FakeRam m{512 * MiB, {GiB, GiB, GiB, GiB}, 1.0, {}};   // short by 4.5 GiB, 4 GiB parked
        const EvictForRam r = run2253(m);
        check(!r.fits && r.evicted == 0 && r.cannot_help && r.short_by == need + floor - 512 * MiB && m.parked.size() == 4,
              "#2253: a shortfall beyond what is parked evicts nothing");
        // the same machine through the loop without the check (base): all four go, refused anyway
        FakeRam old{512 * MiB, {GiB, GiB, GiB, GiB}, 1.0, {}};
        const EvictForRam o = run(old, 4);
        check(!o.fits && o.evicted == 4 && old.parked.empty() && !o.cannot_help, "(without the check: four evictions, an empty cache, refused)");
        // short by exactly what is parked: the evictions can make it, so they run
        FakeRam e{1 * GiB, {GiB, GiB, GiB, GiB}, 1.0, {}};
        const EvictForRam x = run2253(e);
        check(x.fits && x.evicted == 4 && !x.cannot_help, "#2253: short by exactly what is parked: all four go and it fits");
        // already fits: the check never runs
        FakeRam f{8 * GiB, {GiB}, 0.0, {}};
        check(run2253(f).fits && f.parked.size() == 1, "#2253: a park that fits evicts nothing and is not refused");
        // an unknown reading: no figure to compare, the old rule (evict, refuse) stands
        FakeRam u{512 * MiB, {GiB}, 1.0, {}};
        const EvictForRam ur = evict_for_ram([&] { u.last.reset(); return false; }, [&] { return u.last; }, [&] { return u.evict(); }, 4,
                                             need + floor, (uint64_t) 1 * GiB);
        check(!ur.cannot_help && ur.evicted == 1, "#2253: an unknown reading is not taken for a shortfall");
    }
    {   // an unknown reading never fits and never counts as no-progress
        FakeRam m{4 * GiB, {GiB}, 1.0, {}};
        const EvictForRam r = evict_for_ram([&] { m.last.reset(); return false; }, [&] { return m.last; }, [&] { return m.evict(); }, 4);
        check(!r.fits && r.evicted == 1 && !r.no_progress, "an unknown reading: refused (the old rule), not taken for no progress");
    }
}
}

int main() {
    evict_for_ram_cases();
    for (const char* text : {"", "MemFree: 100 kB\n", "MemAvailable: -1 kB\n",
                            "MemAvailable: +1 kB\n", "MemAvailable: 1 MB\n",
                            "MemAvailable: 1\n", "MemAvailable: 1x kB\n",
                            "MemAvailable: 18446744073709551615 kB\n",
                            "MemAvailable: 18446744073709551616 kB\n",
                            "MemAvailable: 1 kB trailing\n",
                            "MemAvailable: 1 kB\nMemAvailable: 2 kB\n"}) {
        std::istringstream input(text);
        check(!conversation_mem_available(input), "missing/malformed telemetry is unknown");
    }
    std::istringstream normal("MemTotal: 999999 kB\nMemAvailable:    12345 kB\nSwapFree: 777 kB\n");
    check(conversation_mem_available(normal) == 12345ULL * 1024, "only MemAvailable is counted");
    std::istringstream zero("MemAvailable: 0 kB");
    check(conversation_mem_available(zero) == 0, "zero available memory is known");
    std::istringstream broken("MemAvailable: 123 kB\n");
    broken.setstate(std::ios::badbit);
    check(!conversation_mem_available(broken), "I/O failure declines admission");
    check(!conversation_memory_admit({}, 0, 0), "unknown fails closed even with zero floor");
    check(conversation_memory_admit(100, 40, 60), "exact allocation plus floor fits");
    check(!conversation_memory_admit(99, 40, 60), "one byte below required memory rejected");
    check(!conversation_memory_admit(59, 0, 60), "floor subtraction cannot underflow");
    check(conversation_memory_admit(60, 0, 60), "post-capture floor check");
    check(!conversation_memory_admit(100, std::numeric_limits<uint64_t>::max(), 1), "allocation arithmetic cannot overflow");
    check(conversation_memory_admit(std::numeric_limits<uint64_t>::max(),
                                   std::numeric_limits<uint64_t>::max(), 0), "maximal exact bound");
    check(!conversation_memory_admit(std::numeric_limits<uint64_t>::max(),
                                    std::numeric_limits<uint64_t>::max(), 1), "maximal sum overflow rejected");
    // #3493: the budget used when --conversation-cache-mib is not given
    const auto GiB = [](uint64_t n) { return n << 30; };
    check(auto_conversation_cache_mib({}, 2560) == 0, "unknown telemetry: no cache");
    check(auto_conversation_cache_mib(GiB(2), 2560) == 0, "below the parking floor: off");
    check(auto_conversation_cache_mib(GiB(4), 2560) == 0, "4 GiB free: 40% of 1.5 GiB is under 1 GiB: off");
    check(auto_conversation_cache_mib(GiB(8), 2560) == 2048, "8 GiB free: 40% of 5.5 GiB, in steps of 256");
    check(auto_conversation_cache_mib(GiB(20), 2560) == 7168, "20 GiB free: 40% of 17.5 GiB");
    check(auto_conversation_cache_mib(GiB(40), 2560) == 8192, "capped at 8 GiB");
    check(auto_conversation_cache_mib(GiB(1000), 0) == 8192, "a big server: still the cap");
    check(auto_conversation_cache_mib(GiB(40), 1000000) == 0, "a floor above the RAM: off, no underflow");
    // Test the real provider without assuming any particular amount of free RAM.
    const auto available = available_host_bytes();
    check(!available || conversation_memory_admit(available, 0, 0), "provider returns bytes or unknown");
    // #4518: after the load, does the PLE table's lock leave the automatic conversation cache no room?  (min free 2560 MiB,
    // cache 40% of the rest, 8192 MiB at most, 1024 MiB at least)
    {
        const uint64_t table = GiB(26) + 870ull * (1ull << 20);   // IQ3_S's table, 26.8 GiB
        const int64_t floor_mib = 2560;
        check(!ple_lock_blocks_cache(std::nullopt, table, floor_mib), "unknown telemetry: leave the lock");
        check(!ple_lock_blocks_cache(GiB(40), 0, floor_mib), "unknown table size: leave the lock");
        check(ple_lock_blocks_cache(GiB(4), table, floor_mib), "4 GiB left with the table locked (cache off), 30 GiB without");
        check(ple_lock_blocks_cache(GiB(10), table, floor_mib), "10 GiB left: cache 1.9 GiB with the lock, 8 GiB without");
        check(!ple_lock_blocks_cache(GiB(40), table, floor_mib), "40 GiB left: the cache is already at its 8 GiB cap");
        check(!ple_lock_blocks_cache(GiB(23), table, floor_mib), "23 GiB: the cap is just reached with the lock");
        check(ple_lock_blocks_cache(GiB(22), table, floor_mib), "22 GiB: the lock costs the cache some of its 8 GiB");
        check(!ple_lock_blocks_cache(GiB(1), 1ull << 20, floor_mib), "a tiny table cannot make a cache");
        check(!ple_lock_blocks_cache(std::numeric_limits<uint64_t>::max(), table, floor_mib), "no overflow");
    }
    std::printf("conversation_memory_test: %d checks passed\n", checks);
}
