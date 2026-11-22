// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// host_available_memory() on fixture files (upstream's test_host_memory, the cases XeStrata's probe has): MemAvailable,
// the cgroup v2 limits above the process, a group without a memory controller and a path outside the mount (#5196), a
// hybrid host's "0::" line (upstream cfb3d9bcb), and cgroup v1's limit.  Host only: no queue and no GPU.
//   host_memory_test
#include "strata/core/expert_source.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

#if defined(__linux__)
#include <unistd.h>
#endif

namespace fs = std::filesystem;

namespace {

int failed = 0;

void require(bool ok, const char* what) {
    if (ok) return;
    std::fprintf(stderr, "host_memory_test: FAILED: %s\n", what);
    ++failed;
}

}  // namespace

int main() {
    using namespace strata::core::detail;
    HostMemory m;
#if defined(__linux__)
    constexpr uint64_t GiB = 1ull << 30u;
    const fs::path dir = fs::temp_directory_path() / ("xestrata_host_memory_test_" + std::to_string((long long) getpid()));
    fs::remove_all(dir);
    auto put = [&](const fs::path& rel, const std::string& text) {
        fs::create_directories((dir / rel).parent_path());
        std::ofstream(dir / rel) << text;
    };
    put("meminfo", "MemTotal: 134217728 kB\nMemAvailable: 104857600 kB\n");   // 100 GiB available
    const std::string mi = (dir / "meminfo").string(), cg = (dir / "cgroup").string(), root = (dir / "fs").string();
    auto probe = [&](const std::string& self) {
        put("cgroup", self);
        m = HostMemory{};
        return host_available_memory(m, mi, cg, root);
    };
    fs::create_directories(dir / "fs");
    require(probe("") && m.available == 100 * GiB && m.cgroup_limit == ~0ull, "no cgroup: MemAvailable alone");
    // v2, a 48 GiB limit with 16 GiB charged, 4 GiB of it clean cache
    put("fs/cgroup.controllers", "memory\n");
    put("fs/box/memory.max", std::to_string(48 * GiB) + "\n");
    put("fs/box/memory.current", std::to_string(16 * GiB) + "\n");
    put("fs/box/memory.stat", "inactive_file " + std::to_string(4 * GiB) + "\nfile_dirty 0\nfile_writeback 0\n");
    require(probe("0::/box\n") && m.cgroup_limit == 48 * GiB && m.available > 32 * GiB && m.available < 48 * GiB, "v2 limit");
    put("fs/box/memory.max", "max\n");
    require(probe("0::/box\n") && m.available == 100 * GiB && m.cgroup_limit == ~0ull, "v2 max");
    put("fs/box/memory.max", "48G\n");
    require(!probe("0::/box\n") && m.why != nullptr, "v2 malformed limit accepted");
    fs::remove(dir / "fs/box/memory.max");
    // #5196: a group without a memory.max has no memory controller and no limit of its own; with no limit above it
    // either, MemAvailable alone
    require(probe("0::/box\n") && m.available == 100 * GiB && m.cgroup_limit == ~0ull,
            "v2 group without memory.max: no limit of its own");
    // ... and the parent's limit still applies to it (a 40 GiB group above a child that has no controller)
    put("fs/memory.max", std::to_string(40 * GiB) + "\n");
    put("fs/memory.current", std::to_string(10 * GiB) + "\n");
    put("fs/memory.stat", "inactive_file " + std::to_string(2 * GiB) + "\nfile_dirty 0\nfile_writeback 0\n");
    require(probe("0::/box\n") && m.cgroup_limit == 40 * GiB && m.available < 100 * GiB,
            "v2: the controllerless child reads its parent's limit");
    // a container that has no cgroup namespace: /proc/self/cgroup names the host's path, /sys/fs/cgroup is its own
    // group (the mount's root)
    require(probe("0::/system.slice/docker-abc.scope\n") && m.cgroup_limit == 40 * GiB && m.available < 100 * GiB,
            "v2: a path that is not in the mount reads the mount's own group");
    fs::remove(dir / "fs/memory.max");
    fs::remove(dir / "fs/memory.current");
    fs::remove(dir / "fs/memory.stat");
    // an unreadable limit (a directory where the file should be) is still not guessed at, and says what it could not read
    fs::create_directories(dir / "fs/box/memory.max");
    require(!probe("0::/box\n") && m.why != nullptr, "v2 unreadable memory.max accepted");
    fs::remove_all(dir / "fs/box/memory.max");
    // v1: the memory controller's group, a 32 GiB limit with 10 GiB used
    put("fs/memory/docker/abc/memory.limit_in_bytes", std::to_string(32 * GiB) + "\n");
    put("fs/memory/docker/abc/memory.usage_in_bytes", std::to_string(10 * GiB) + "\n");
    put("fs/memory/memory.limit_in_bytes", "9223372036854771712\n");   // the root: unlimited
    put("fs/memory/memory.usage_in_bytes", std::to_string(50 * GiB) + "\n");
    require(probe("12:cpu,cpuacct:/docker/abc\n4:memory:/docker/abc\n1:name=systemd:/docker/abc\n") &&
            m.available == 22 * GiB && m.cgroup_limit == 32 * GiB, "v1 limit");
    // a hybrid host: v1 controllers plus a "0::/..." line, and no cgroup.controllers at the root (the root is the v1
    // tree, the "0::" path is not in it).  The v1 limit must apply (it was "cannot determine")
    fs::remove(dir / "fs/cgroup.controllers");
    const std::string hybrid = "12:cpu,cpuacct:/docker/abc\n4:memory:/docker/abc\n1:name=systemd:/docker/abc\n0::/docker/abc\n";
    require(probe(hybrid) && m.available == 22 * GiB && m.cgroup_limit == 32 * GiB, "hybrid: the v1 limit applies");
    require(probe("0::/docker/abc\n4:memory:/docker/abc\n") && m.available == 22 * GiB, "hybrid: the 0:: line first");
    require(probe("4:memory:/docker/abc\n0::/\n") && m.available == 22 * GiB, "hybrid: a 0::/ line");
    put("fs/memory/docker/abc/memory.limit_in_bytes", "9223372036854771712\n");
    require(probe(hybrid) && m.available == 100 * GiB && m.cgroup_limit == ~0ull, "hybrid: v1 unlimited");
    require(probe("0::/docker/abc\n") && m.available == 100 * GiB,
            "a 0:: line with no controllers file and no v1 line: MemAvailable alone");
    // with cgroup.controllers (a real v2 root) the 0:: line is the limit and a v1 line beside it is not read
    put("fs/cgroup.controllers", "memory\n");
    put("fs/box/memory.max", std::to_string(48 * GiB) + "\n");
    put("fs/memory/docker/abc/memory.limit_in_bytes", std::to_string(32 * GiB) + "\n");
    require(probe("4:memory:/docker/abc\n0::/box\n") && m.cgroup_limit == 48 * GiB, "v2 root: the 0:: line wins over a v1 line");
    require(probe("0::/nowhere\n") && m.available == 100 * GiB && m.cgroup_limit == ~0ull,
            "v2 root: a 0:: path that is not a directory: the mount's own group");
    fs::remove_all(dir);
#else
    require(host_available_memory(m) && m.available > 0, "this PC's RAM");
#endif
    if (failed) return 1;
    std::printf("host_memory_test: OK\n");
    return 0;
}
