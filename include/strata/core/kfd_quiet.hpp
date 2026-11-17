// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
//
// Keeping the KFD quiet while our GPU queues are busy, on an integrated AMD GPU under Linux (upstream f9932c6ae, 6069c9d0c,
// f73fa7410, 0ee263992).
//
// Radeon 780M (gfx1103), Linux, upstream's p2006: the GPU resets when the kernel evicts the process's queues while a dispatch is
// in flight.  The eviction comes from page moves inside memory the GPU reaches as a userptr (a registered expert arena; pageable
// copies the HIP runtime pins in place).  What the engine controls is WHEN it provokes the kernel (its own allocations and copies)
// and that its queues are idle then.  The pieces XeStrata takes, one switch each:
//   LOAD  one more KFD settle (kfd_settle.hpp) between the expert cache's allocation and its fill; the CPU prompt share's pinned
//         buffers made in Prefill::init instead of at the first request's first sharing layer; one staged copy each way before the
//         last settle, so the runtime makes its staging buffer during the load;
//   XFER  GPU_PINNED_MIN_XFER_SIZE set before the HIP runtime starts (pageable copies go through the runtime's staging buffers
//         instead of pinning our pages in place); a value the caller set wins.
// Upstream's hyper-connection check before the arena (XeStrata has no fused_gr_check), its first prompt's lazy device buffers
// (XeStrata's prompt path carves them in Prefill::init already) and its EXIT order (XeStrata's route_stop_signals drains the
// queues before it ends) are not here.  None changes a kernel or what is computed.  STRATA_KFD_QUIET=0 turns all of them off
// (A/B with the same binary); STRATA_KFD_QUIET_LOAD / _XFER = 0 turn one off.  STRATA_TRACE prints one line per change that is
// active.
//
// The class: upstream's is gfx1103 / gfx1150 / gfx1151 by name; XeStrata's is what the device reports: every engine GPU an
// integrated one on the HIP backend, under Linux (core::kfd_igpu_class, device.cpp), and before HIP starts every GPU node of the
// KFD topology without memory of its own (local_mem_size 0: its memory is the system RAM).  Radeon 890M, Linux 7.2: its node reads
// local_mem_size 0 beside a 512 MiB BIOS carve-out; a discrete card reports its VRAM there (not checked on one here).
//
// GIVE BACK (upstream 0ee263992, 6d8e42886): after the load the page cache of the model files the engine copied from (the pack's
// dense.bin and embd.bin, an arena's expert files, the GGUF shards of --native, never the PLE table's) is given back (platform::drop_file_cache), so the
// kernel's proactive compaction finds free memory instead of moving pages; the files the expert source still reads in place
// (--mmap-experts: experts.bin or the GGUF shards) keep theirs.  The experts' own cache is given back as the host USM cache
// fills (ExpertSource::release, generate.cpp), as before.  STRATA_NO_ARENA_CACHE_DROP=1 keeps them.
//
// Pure pieces, testable without a GPU (tests/core/kfd_quiet_test.cpp).
#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

#include <cstdio>

#if defined(__linux__)
#include <dirent.h>
#endif

namespace strata::core::kfd_quiet {

enum class Fix { Load, Xfer };

/// The device class, set by the engine once the GPU is named (core::kfd_igpu_class).  Load does nothing while it is false.
inline std::atomic<bool>& class_flag() {
    static std::atomic<bool> f{false};
    return f;
}

/// The switch rule: `all` is STRATA_KFD_QUIET, `one` the fix's own variable (nullptr = unset).  A value starting with '0' turns off;
/// unset, empty or anything else leaves it on.
inline bool switch_on(const char* all, const char* one) {
    auto off = [](const char* v) { return v != nullptr && v[0] == '0'; };
    return !off(all) && !off(one);
}

inline const char* fix_env(Fix f) { return f == Fix::Xfer ? "STRATA_KFD_QUIET_XFER" : "STRATA_KFD_QUIET_LOAD"; }

inline bool env_on(Fix f) { return switch_on(std::getenv("STRATA_KFD_QUIET"), std::getenv(fix_env(f))); }

/// Load: the device class and the switches.
inline bool on(Fix f) { return class_flag().load(std::memory_order_relaxed) && env_on(f); }

/// LOAD, as one rule for its call sites (generate.cpp: the settle after the expert cache, the staged copy before READY;
/// prefill.cpp: the CPU share's buffers in Prefill::init).  `cls`: class_flag(); `all` / `one`: the switches.
struct LoadMoves {
    bool settle_after_cache = false;
    bool cpu_share_buffers = false;
    bool staged_copy_warm = false;
};
inline LoadMoves load_moves(bool cls, const char* all, const char* one) {
    LoadMoves m;
    m.settle_after_cache = m.cpu_share_buffers = m.staged_copy_warm = cls && switch_on(all, one);
    return m;
}
inline LoadMoves load_moves() {
    return load_moves(class_flag().load(std::memory_order_relaxed), std::getenv("STRATA_KFD_QUIET"), std::getenv(fix_env(Fix::Load)));
}

/// The staged copy before READY: bytes per direction.  More than the 1 MiB below which the runtime always stages a pageable copy,
/// so the copy takes the path a larger pageable copy takes with GPU_PINNED_MIN_XFER_SIZE set (XFER).
inline constexpr size_t kStagedWarmBytes = (size_t) 2 * 1024 * 1024;

// ---- XFER: the device class from the KFD topology (before HIP starts) ------------------------------------------------------------

/// One node's properties text: `simd_count` (0 when absent: a CPU node) and `local_mem_size` (the GPU's own memory, bytes).
struct Node {
    int64_t simds = 0;
    int64_t local_mem = 0;
    bool has_local_mem = false;   ///< the key is there (a kernel without it does not say, and is not the class)
};

inline Node parse_node_properties(const std::string& text) {
    Node n;
    size_t i = 0;
    while (i < text.size()) {
        size_t e = text.find('\n', i);
        if (e == std::string::npos) e = text.size();
        const std::string line = text.substr(i, e - i);
        i = e + 1;
        const size_t sp = line.find(' ');
        if (sp == std::string::npos) continue;
        const std::string key = line.substr(0, sp);
        const bool simd = key == "simd_count", mem = key == "local_mem_size";
        if (!simd && !mem) continue;
        int64_t v = 0;
        size_t j = sp;
        while (j < line.size() && line[j] == ' ') ++j;
        bool any = false;
        for (; j < line.size() && line[j] >= '0' && line[j] <= '9' && v < (int64_t) 100000000000000000LL; ++j) {
            v = v * 10 + (line[j] - '0');
            any = true;
        }
        if (!any) continue;
        if (simd) n.simds = v;
        else { n.local_mem = v; n.has_local_mem = true; }
    }
    return n;
}

/// The class: at least one GPU node, and every GPU node one without memory of its own (local_mem_size present and 0).  A machine
/// with a discrete card beside the integrated GPU is not the class, so its path stays the old one.
inline bool nodes_in_class(const std::vector<Node>& nodes) {
    int gpus = 0;
    for (const Node& n : nodes) {
        if (n.simds <= 0) continue;
        ++gpus;
        if (!n.has_local_mem || n.local_mem != 0) return false;
    }
    return gpus > 0;
}

#if defined(__linux__)
/// The nodes under `topo` (normally /sys/class/kfd/kfd/topology/nodes); empty when the folder is not there.
inline std::vector<Node> read_topology(const std::string& topo) {
    std::vector<Node> out;
    DIR* d = opendir(topo.c_str());
    if (d == nullptr) return out;
    while (const dirent* e = readdir(d)) {
        if (e->d_name[0] < '0' || e->d_name[0] > '9') continue;
        const std::string path = topo + "/" + e->d_name + "/properties";
        std::FILE* f = std::fopen(path.c_str(), "r");
        if (f == nullptr) continue;
        std::string text;
        char buf[4096];
        for (;;) {   // to the end of the file or the first error: a short read ends it
            const size_t got = std::fread(buf, 1, sizeof buf, f);
            text.append(buf, got);
            if (got < sizeof buf) break;
        }
        std::fclose(f);
        out.push_back(parse_node_properties(text));
    }
    closedir(d);
    return out;
}
#endif

/// XFER: the value the engine sets, or nullptr when nothing is to be set.  `in_class`: the topology rule; `already`:
/// GPU_PINNED_MIN_XFER_SIZE as the caller set it (a value the caller set wins).  1000000 is upstream's (and its STRATA_ARENA_MMAP's):
/// the runtime reads it in MiB, so no pageable copy of ours reaches the pin-in-place path.
inline const char* xfer_value(bool in_class, const char* all, const char* one, const char* already) {
    if (!in_class || !switch_on(all, one) || already != nullptr) return nullptr;
    return "1000000";
}

// ---- GIVE BACK: the model files' page cache after the load -----------------------------------------------------------------------

/// STRATA_NO_ARENA_CACHE_DROP: anything but unset, empty or 0 keeps the cache.
inline bool cache_drop_on(bool cls, const char* no_drop) {
    return cls && !(no_drop != nullptr && no_drop[0] != '\0' && no_drop[0] != '0');
}

/// The files whose cache is given back: the pack's dense.bin and embd.bin (`pack` empty: none), the GGUF shards of --native and the
/// files the expert source read (`expert_files`, ExpertSource::model_inputs), never the PLE table's (its rows are read from the drive
/// on request).  When `experts_in_place` (--mmap-experts: the expert source still reads the model's files, the GGUF shards among
/// them) no shard and none of `expert_files` is given back; an arena copied its experts and reads them no more.  Empty and repeated
/// paths are left out.
inline std::vector<std::string> cache_drop_files(const std::string& pack, const std::vector<std::string>& native_shards,
                                                 const std::string& ple_gguf, bool experts_in_place,
                                                 const std::vector<std::string>& expert_files) {
    std::vector<std::string> want;
    if (!pack.empty())
        for (const char* n : {"dense.bin", "embd.bin"}) want.push_back(pack + "/" + n);
    if (!experts_in_place) {
        want.insert(want.end(), expert_files.begin(), expert_files.end());
        want.insert(want.end(), native_shards.begin(), native_shards.end());
    }
    std::vector<std::string> out;
    for (const std::string& p : want) {
        if (p.empty() || p == ple_gguf || std::find(out.begin(), out.end(), p) != out.end()) continue;
        if (experts_in_place && std::find(expert_files.begin(), expert_files.end(), p) != expert_files.end()) continue;
        out.push_back(p);
    }
    return out;
}

}  // namespace strata::core::kfd_quiet
