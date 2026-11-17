// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
//
// keeping the KFD quiet on an integrated AMD GPU under Linux (include/strata/core/kfd_quiet.hpp; upstream f9932c6ae, 6069c9d0c,
// f73fa7410, 0ee263992).  No GPU: the switch rule, the load order's rule, the class from the KFD topology (XeStrata: the GPU nodes
// without memory of their own, not upstream's gfx names), the staged-copy value, the files whose cache is given back, and the
// arena's huge-page rule (pinned.hpp, upstream 0a17fb99e).
#include "strata/core/kfd_quiet.hpp"
#include "strata/core/pinned.hpp"

#include <cstdio>
#include <string>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <sys/stat.h>
#include <unistd.h>
#include <cstdlib>
#endif

using namespace strata::core::kfd_quiet;

namespace {
int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

void switches() {
    CHECK(switch_on(nullptr, nullptr));
    CHECK(switch_on("1", nullptr));
    CHECK(switch_on("", ""));            // empty = unset
    CHECK(!switch_on("0", nullptr));     // the kill switch turns everything off
    CHECK(!switch_on("0", "1"));         // ... also when the fix's own switch says 1
    CHECK(!switch_on(nullptr, "0"));     // one fix off
    CHECK(!switch_on("1", "0"));
    CHECK(switch_on("yes", "on"));
    CHECK(std::string(fix_env(Fix::Load)) == "STRATA_KFD_QUIET_LOAD");
    CHECK(std::string(fix_env(Fix::Xfer)) == "STRATA_KFD_QUIET_XFER");
}

void load_order() {
    // not the class: none of the moves, whatever the switches say (every other GPU keeps the old order)
    for (const char* all : {(const char*) nullptr, "1", "0"})
        for (const char* one : {(const char*) nullptr, "1", "0"}) {
            const LoadMoves m = load_moves(false, all, one);
            CHECK(!m.settle_after_cache && !m.cpu_share_buffers && !m.staged_copy_warm);
        }
    // the class, nothing set: all of them
    {
        const LoadMoves m = load_moves(true, nullptr, nullptr);
        CHECK(m.settle_after_cache && m.cpu_share_buffers && m.staged_copy_warm);
    }
    CHECK(kStagedWarmBytes > (size_t) 1024 * 1024);   // above the runtime's own staging threshold
    // the class, STRATA_KFD_QUIET=0 or STRATA_KFD_QUIET_LOAD=0: the old order (the A/B arm of the same binary)
    using P = std::pair<const char*, const char*>;
    for (const P& p : std::vector<P>{P{"0", nullptr}, P{nullptr, "0"}, P{"0", "0"}, P{"0", "1"}}) {
        const LoadMoves m = load_moves(true, p.first, p.second);
        CHECK(!m.settle_after_cache && !m.cpu_share_buffers && !m.staged_copy_warm);
    }
    // the flag as the engine sets it: off until the GPU is named
    CHECK(!class_flag().load());
    CHECK(!on(Fix::Load));
}

// the class from the topology text, as /sys/class/kfd/kfd/topology/nodes/N/properties reads (shortened).  The 890M's two nodes are
// this machine's (Linux 7.2); the discrete ones carry their VRAM in local_mem_size.
const char* kCpuNode = "cpu_cores_count 24\nsimd_count 0\nmem_banks_count 1\ngfx_target_version 0\nvendor_id 0\n";
const char* k890M = "cpu_cores_count 0\nsimd_count 32\nmem_banks_count 1\ngfx_target_version 110500\nvendor_id 4098\n"
                    "device_id 5390\nlocal_mem_size 0\nfw_version 36\n";
const char* k780M = "simd_count 24\r\ngfx_target_version 110003\r\nlocal_mem_size 0\r\n";   // \r tolerated: the number ends there
const char* k7900 = "simd_count 192\ngfx_target_version 110000\nlocal_mem_size 25753026560\n";   // a discrete RX 7900 XTX
const char* kR9700 = "simd_count 128\ngfx_target_version 120001\nlocal_mem_size 34342961152\n";
const char* kNoKey = "simd_count 32\ngfx_target_version 110500\n";   // a kernel that does not say: not the class

void topology() {
    {
        const Node n = parse_node_properties(k890M);
        CHECK(n.simds == 32 && n.local_mem == 0 && n.has_local_mem);
        const Node c = parse_node_properties(kCpuNode);
        CHECK(c.simds == 0 && !c.has_local_mem);
        CHECK(parse_node_properties(k7900).local_mem == 25753026560LL);
        CHECK(parse_node_properties(k780M).has_local_mem);
        CHECK(!parse_node_properties("").has_local_mem);
        CHECK(parse_node_properties("local_mem_size\nsimd_count x\n").simds == 0);       // no number: not read
        CHECK(!parse_node_properties("local_mem_size x\n").has_local_mem);
        CHECK(!parse_node_properties("xlocal_mem_size 0\n").has_local_mem);              // another key
    }
    auto cls = [](const std::vector<const char*>& nodes) {
        std::vector<Node> v;
        v.reserve(nodes.size());
        for (const char* t : nodes) v.push_back(parse_node_properties(t));
        return nodes_in_class(v);
    };
    CHECK(cls({kCpuNode, k890M}));
    CHECK(cls({kCpuNode, k780M}));
    CHECK(!cls({}));                        // no topology (no KFD, another OS, a container without /sys/class/kfd)
    CHECK(!cls({kCpuNode}));                // no GPU node
    CHECK(!cls({kCpuNode, k7900}));         // a discrete card
    CHECK(!cls({kCpuNode, kR9700}));
    CHECK(!cls({kCpuNode, kNoKey}));        // the kernel does not say whether it has memory of its own
    CHECK(!cls({kCpuNode, k890M, k7900}));  // an 890M beside a discrete card: not the class (the card keeps its path)
    // the value: only in the class, only with the switches on, never over the caller's own value
    CHECK(xfer_value(true, nullptr, nullptr, nullptr) != nullptr && std::string(xfer_value(true, nullptr, nullptr, nullptr)) == "1000000");
    CHECK(xfer_value(false, nullptr, nullptr, nullptr) == nullptr);
    CHECK(xfer_value(true, "0", nullptr, nullptr) == nullptr);
    CHECK(xfer_value(true, nullptr, "0", nullptr) == nullptr);
    CHECK(xfer_value(true, nullptr, nullptr, "4") == nullptr);
    CHECK(xfer_value(true, nullptr, nullptr, "") == nullptr);   // set, even empty: the caller's
#if defined(__linux__)
    // the folder reader on a temporary tree: two nodes, a stray file, a node without properties
    char tmpl[] = "/tmp/kfdqXXXXXX";
    if (mkdtemp(tmpl) != nullptr) {
        const std::string root = tmpl;
        auto put = [](const std::string& path, const char* text) {
            if (std::FILE* f = std::fopen(path.c_str(), "w")) { std::fputs(text, f); std::fclose(f); }
        };
        mkdir((root + "/0").c_str(), 0700);
        mkdir((root + "/1").c_str(), 0700);
        mkdir((root + "/2").c_str(), 0700);
        put(root + "/0/properties", kCpuNode);
        put(root + "/1/properties", k890M);
        put(root + "/README", "not a node");
        const std::vector<Node> v = read_topology(root);
        CHECK(v.size() == 2);
        CHECK(nodes_in_class(v));
        put(root + "/2/properties", k7900);
        CHECK(!nodes_in_class(read_topology(root)));
        CHECK(read_topology(root + "/none").empty());
        for (const char* p : {"/0/properties", "/1/properties", "/2/properties", "/README"}) std::remove((root + p).c_str());
        for (const char* p : {"/0", "/1", "/2", ""}) rmdir((root + p).c_str());
    } else {
        std::fprintf(stderr, "FAIL %s:%d: mkdtemp failed\n", __FILE__, __LINE__);
        ++failed;
    }
    // this machine's own topology: whatever it is, the call returns
    (void) nodes_in_class(read_topology("/sys/class/kfd/kfd/topology/nodes"));
#endif
}

void give_back() {
    // the switch: only in the class; STRATA_NO_ARENA_CACHE_DROP set (not empty, not 0) keeps the cache
    CHECK(cache_drop_on(true, nullptr));
    CHECK(cache_drop_on(true, ""));
    CHECK(cache_drop_on(true, "0"));
    CHECK(!cache_drop_on(true, "1"));
    CHECK(!cache_drop_on(false, nullptr));
    using V = std::vector<std::string>;
    // an arena (the experts copied): the pack's two files, its expert files and every shard but the PLE table's
    CHECK(cache_drop_files("/p", V{"/m-1.gguf", "/m-2.gguf"}, "/m-1.gguf", false, V{"/p/experts.bin"}) ==
          (V{"/p/dense.bin", "/p/embd.bin", "/p/experts.bin", "/m-2.gguf"}));
    CHECK(cache_drop_files("/p", V{"/m-1.gguf", "/m-2.gguf"}, "", false, V{}) ==
          (V{"/p/dense.bin", "/p/embd.bin", "/m-1.gguf", "/m-2.gguf"}));
    CHECK(cache_drop_files("/p", V{"/m-1.gguf", "/m-2.gguf"}, "/m-1.gguf", false, V{"/m-1.gguf", "/m-2.gguf"}) ==
          (V{"/p/dense.bin", "/p/embd.bin", "/m-2.gguf"}));   // an arena read from the shards: once each, never the PLE one
    // --mmap-experts: the expert source reads the model's files in place; no shard, and none of the files it names
    CHECK(cache_drop_files("/p", V{"/m-1.gguf", "/m-2.gguf"}, "/m-1.gguf", true, V{"/m-2.gguf"}) ==
          (V{"/p/dense.bin", "/p/embd.bin"}));
    CHECK(cache_drop_files("/p", V{}, "", true, V{"/p/experts.bin"}) == (V{"/p/dense.bin", "/p/embd.bin"}));
    CHECK(cache_drop_files("/p", V{}, "", true, V{"/p/dense.bin"}) == (V{"/p/embd.bin"}));
    // no pack, repeated and empty paths
    CHECK(cache_drop_files("", V{"/a.gguf", "", "/a.gguf"}, "", false, V{}) == (V{"/a.gguf"}));
    CHECK(cache_drop_files("", V{}, "", false, V{}).empty());
}

void arena_thp() {
    using strata::core::arena_skip_thp_request;
    // STRATA_NO_ARENA_THP: unset = the device rule decides; "0" = ask for the huge pages (also in the class); any other value = skip
    CHECK(arena_skip_thp_request(nullptr, false, true));      // the integrated GPU, nothing set: skipped
    CHECK(!arena_skip_thp_request(nullptr, false, false));    // any other GPU, nothing set: requested (as before)
    CHECK(!arena_skip_thp_request("0", false, true));         // the user asks for them on the integrated GPU
    CHECK(!arena_skip_thp_request("0", false, false));
    CHECK(arena_skip_thp_request("1", false, false));         // =1 skips on any GPU (as before)
    CHECK(arena_skip_thp_request("1", false, true));
    CHECK(arena_skip_thp_request("", false, false));          // set but empty: skipped, as before
    CHECK(arena_skip_thp_request(nullptr, true, false));      // THP "always": never requested (upstream #771)
    CHECK(arena_skip_thp_request("0", true, true));
}
}  // namespace

int main() {
    switches();
    load_order();
    topology();
    give_back();
    arena_thp();
    if (failed) {
        std::fprintf(stderr, "kfd_quiet_test: %d check(s) failed\n", failed);
        return 1;
    }
    std::printf("kfd_quiet_test: ok\n");
    return 0;
}
