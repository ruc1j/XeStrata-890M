// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
//
// a short wait before READY on an integrated AMD GPU under Linux, until the kernel has stopped moving the process's registered
// memory (upstream 8f6393a37).
//
// Radeon 780M (gfx1103), Linux, expert arena registered with the driver (a KFD userptr).  The load that fills the 31 GiB arena and the
// GPU buffers keeps the kernel moving pages (compaction, reclaim) for about two seconds after the engine is up; every move inside a
// registered range makes KFD evict the process's GPU queues, and the gc 11 MES firmware never answers the eviction while a dispatch
// sits in a barrier: the driver resets the GPU and the first prompt ends in "unspecified launch failure" (60 s hang).  p2006, g++ load
// beside the start, registered arena, 2K + 8K prompts: first prompt right after READY 5 of 10 failed in its first seconds; the first
// request 30 s later 0 of 5, 2 s after the process's own eviction counter went quiet 0 of 10.  The counter moves
// until about READY + 2 s and never after.
//
// So before it prints READY the engine waits until its own KFD eviction counter (/sys/class/kfd/kfd/proc/<pid>/stats_*/evicted_ms,
// milliseconds the process's queues spent evicted) has not moved for 2 s, at most 10 s.  Where the counter cannot be read (no such file,
// another kernel, a container with its own pid namespace) it waits nothing and says nothing.  Automatic on the device class (below);
// STRATA_NO_KFD_SETTLE=1 turns it off (research / A-B only).
//
// XeStrata-890M: the class is every engine GPU an integrated one on the HIP backend under Linux, as the device reports it
// (core::kfd_igpu_class, src/core/device.cpp), not upstream's list of gfx names; the Radeon 890M's failures were not
// measured, the 780M's are upstream's.  The engine prints the time it waited (generate.cpp), so a slower READY shows why.
//
// Pure pieces, so that they can be tested without a GPU: parse_evicted_ms (the text of one counter file), read_own_evicted_ms (the
// folder of one process), settle (the wait, with the reader and the clock passed in).
#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <thread>

#include <cstdio>

#if defined(__linux__)
#include <dirent.h>
#include <unistd.h>
#endif

namespace strata::core {

/// Set by the engine on the device class this is for (core::kfd_igpu_class: every engine GPU integrated, HIP, Linux);
/// settle_before_ready does nothing while it is false.
inline std::atomic<bool>& kfd_settle_flag() {
    static std::atomic<bool> f{false};
    return f;
}

namespace kfd_settle {

struct Reading {
    enum Kind { Missing, Unreadable, Ok } kind = Missing;   // Missing: no counter folder / file; Unreadable: there, but not a number
    int64_t ms = 0;                                          // the sum over the process's GPUs, milliseconds
};

/// The text of one evicted_ms file: a non-negative decimal number, whitespace around it allowed.  Empty text, signs, letters, a second
/// number or more than 18 digits are not a counter.
inline bool parse_evicted_ms(const std::string& text, int64_t& ms) {
    auto space = [](char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; };
    size_t i = 0;
    const size_t n = text.size();
    while (i < n && space(text[i])) ++i;
    const size_t first = i;
    int64_t v = 0;
    while (i < n && text[i] >= '0' && text[i] <= '9') {
        if (i - first >= 18) return false;
        v = v * 10 + (text[i] - '0');
        ++i;
    }
    if (i == first) return false;
    while (i < n && space(text[i])) ++i;
    if (i != n) return false;
    ms = v;
    return true;
}

#if defined(__linux__)
/// The folder KFD keeps for this process.
inline std::string own_proc_dir() { return "/sys/class/kfd/kfd/proc/" + std::to_string((long long) getpid()); }

/// The process's eviction counter: the sum of <proc_dir>/stats_*/evicted_ms.  Missing when the folder is not there or holds no such file,
/// Unreadable when a file is there and cannot be read or does not hold a number.
inline Reading read_own_evicted_ms(const std::string& proc_dir) {
    Reading r;
    DIR* d = opendir(proc_dir.c_str());
    if (d == nullptr) return r;
    int64_t sum = 0;
    bool any = false, bad = false;
    while (const dirent* e = readdir(d)) {
        const std::string name = e->d_name;
        if (name.compare(0, 6, "stats_") != 0) continue;
        std::string path = proc_dir;
        path += '/';
        path += name;
        path += "/evicted_ms";
        std::FILE* f = std::fopen(path.c_str(), "r");
        if (f == nullptr) continue;   // this GPU has no counter file
        char buf[64] = {0};
        const size_t got = std::fread(buf, 1, sizeof buf - 1, f);
        const bool err = std::ferror(f) != 0;
        std::fclose(f);
        int64_t ms = 0;
        if (err || !parse_evicted_ms(std::string(buf, got), ms)) { bad = true; break; }
        sum += ms;
        any = true;
    }
    closedir(d);
    if (bad) { r.kind = Reading::Unreadable; return r; }
    if (!any) return r;
    r.kind = Reading::Ok;
    r.ms = sum;
    return r;
}
#endif

struct Result {
    enum Why { NoCounter, Quiet, Cap } why = NoCounter;   // NoCounter: nothing to watch (or it vanished); Quiet: not moved for quiet_s; Cap: cap_s reached
    double waited_s = 0.0;
    int64_t ms = 0;                                        // the counter at the end
};

/// The wait.  `read_fn()` -> Reading, `now_fn()` -> seconds (monotonic), `sleep_fn(s)`.  The first reading decides: anything but Ok returns at once
/// (NoCounter, waited 0).  Then it polls every poll_s; the quiet clock starts at the first reading, so the least wait is quiet_s; the cap
/// counts from the first reading too.  A counter that stops being readable ends the wait (NoCounter).  Quiet is judged before the cap.
template <class ReadFn, class NowFn, class SleepFn>
Result settle(ReadFn read_fn, NowFn now_fn, SleepFn sleep_fn, double quiet_s, double cap_s, double poll_s) {
    Result r;
    const Reading first = read_fn();
    if (first.kind != Reading::Ok) return r;
    const double t0 = now_fn();
    double last_move = t0;
    r.ms = first.ms;
    for (;;) {
        sleep_fn(poll_s);
        const Reading x = read_fn();
        const double t = now_fn();
        r.waited_s = t - t0;
        if (x.kind != Reading::Ok) { r.why = Result::NoCounter; return r; }
        if (x.ms != r.ms) { r.ms = x.ms; last_move = t; }
        if (t - last_move >= quiet_s) { r.why = Result::Quiet; return r; }
        if (t - t0 >= cap_s) { r.why = Result::Cap; return r; }
    }
}

/// The engine's call, just before it prints READY (and, kfd_quiet.hpp's LOAD, after the expert cache is allocated).  Does nothing
/// unless the device class set the flag, STRATA_NO_KFD_SETTLE is not set (anything but empty or 0 turns it off), and the counter can
/// be read.  Prints nothing: XeStrata's caller prints `line(result)`, the time waited.
inline Result settle_before_ready() {
#if defined(__linux__)
    if (!kfd_settle_flag().load(std::memory_order_relaxed)) return Result();
    if (const char* e = std::getenv("STRATA_NO_KFD_SETTLE"); e != nullptr && e[0] != '\0' && e[0] != '0') return Result();
    const std::string dir = own_proc_dir();
    using clock = std::chrono::steady_clock;
    const auto base = clock::now();
    return settle([&] { return read_own_evicted_ms(dir); },
                  [&] { return std::chrono::duration<double>(clock::now() - base).count(); },
                  [](double s) { std::this_thread::sleep_for(std::chrono::duration<double>(s)); },
                  /*quiet_s*/ 2.0, /*cap_s*/ 10.0, /*poll_s*/ 0.2);
#else
    return Result();
#endif
}

/// XeStrata: the line the engine prints for a wait (no newline), or "" when nothing was waited (no counter, or not this class).
/// `where`: what the wait comes before ("READY", "the expert cache's fill").
inline std::string line(const Result& r, const char* where) {
    if (r.why == Result::NoCounter && r.waited_s <= 0.0) return std::string();
    char buf[320];
    std::snprintf(buf, sizeof buf,
                  "waited %.1f s before %s until the GPU driver's queue pauses (KFD evictions) had stopped for 2 s (%s; "
                  "%lld ms paused since the start; STRATA_NO_KFD_SETTLE=1: no wait)",
                  r.waited_s, where,
                  r.why == Result::Quiet ? "quiet" : r.why == Result::Cap ? "still moving at the 10 s cap" : "the counter went away",
                  (long long) r.ms);
    return std::string(buf);
}

}  // namespace kfd_settle
}  // namespace strata::core
