// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// The DFlash2 pack loader refuses a pack whose own numbers do not fit the file or the drafter (Issue #78): a tensor
// offset or size outside the file, an index longer than the file, and a layer count, inter size, conv group or tap list
// the drafter's buffers cannot hold.  Each is refused before anything is allocated on the device or copied to it.
#include "strata/core/dflash2.hpp"
#include "strata/core/runtime.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

namespace {

const char* const kPath = "dflash2_pack_test.df2p";

// The drafter's shape (the only one the loader takes), with the line for `key` replaced by `value` (or dropped when
// `value` is empty).
std::string cfg(const std::string& key = "", const std::string& value = "") {
    const char* const lines[][2] = {{"hidden", "2560"}, {"layers", "5"},  {"heads", "32"}, {"kv_heads", "8"},
                                    {"head_dim", "128"}, {"inter", "7680"}, {"window", "2048"}, {"conv_group", "16"},
                                    {"rank", "256"},     {"taps", "1,2,3,4,5"}};
    std::string out;
    for (const auto& l : lines) {
        if (key == l[0]) {
            if (!value.empty()) out += std::string("cfg ") + l[0] + " " + value + "\n";
        } else {
            out += std::string("cfg ") + l[0] + " " + l[1] + "\n";
        }
    }
    return out;
}

// A pack of `index` and `data_bytes` zero bytes after it; `index_len` overrides the header's index length.
void write_pack(const std::string& index, size_t data_bytes, int64_t index_len = -1) {
    std::ofstream f(kPath, std::ios::binary | std::ios::trunc);
    const uint32_t hdr[3] = {0, 1, (uint32_t) (index_len < 0 ? (int64_t) index.size() : index_len)};
    f.write(reinterpret_cast<const char*>(hdr), 12);
    f.seekp(0);
    f.write("DF2P", 4);
    f.seekp(12);
    f.write(index.data(), (std::streamsize) index.size());
    const std::vector<char> zeros(data_bytes, 0);
    f.write(zeros.data(), (std::streamsize) zeros.size());
}

bool refused(const std::string& want) {
    strata::core::Dflash2Drafter d;
    std::string err;
    const bool ok = d.load(kPath, 0, err);
    if (ok || err.find(want) == std::string::npos) {
        std::fprintf(stderr, "  load: %s, error \"%s\" (wanted \"%s\")\n", ok ? "accepted" : "refused", err.c_str(), want.c_str());
        return false;
    }
    return true;
}

}  // namespace

int main() {
    if (strata::core::Runtime::count() < 1) {
        std::printf("dflash2_pack_test: no GPU - skipped\n");
        return 77;
    }
    const std::string t = "t fc.weight f32 1 16 0 64\n";
    // a tensor offset or size the file does not hold
    write_pack(cfg() + "t fc.weight f32 1 16 -8 64\n", 256);
    CHECK(refused("outside the pack"));
    write_pack(cfg() + "t fc.weight f32 1 16 64 -1\n", 256);
    CHECK(refused("outside the pack"));
    write_pack(cfg() + "t fc.weight f32 1 16 128 256\n", 256);
    CHECK(refused("outside the pack"));
    write_pack(cfg() + "t fc.weight f32 1 16 9223372036854775807 9223372036854775807\n", 256);
    CHECK(refused("outside the pack"));
    write_pack(cfg() + "t fc.weight q4_0 4 64 0 100\n", 256);   // 4 x 64 Q4_0 is 144 bytes
    CHECK(refused("Q4_0"));
    // an index longer than the file
    write_pack(cfg() + t, 256, (int64_t) 0xfffffff0);
    CHECK(refused("cut short"));
    // a shape the drafter's buffers cannot hold
    const std::pair<const char*, const char*> shapes[] = {
        {"layers", "0"},      {"layers", "-3"},       {"layers", "9"},       {"inter", "0"},
        {"inter", "-32"},     {"conv_group", "0"},    {"conv_group", "-16"}, {"conv_group", "8"},
        {"taps", ""},         {"taps", "1,2,3,4,5,6"}, {"taps", "1,-2"},     {"window", "0"}};
    for (const auto& [key, value] : shapes) {
        write_pack(cfg(key, value) + t, 256);
        if (!refused("shape")) {
            std::fprintf(stderr, "  with %s %s\n", key, value);
            ++fails;
        }
    }
    std::remove(kPath);
    if (fails) return 1;
    std::printf("dflash2_pack_test: ok\n");
    return 0;
}
