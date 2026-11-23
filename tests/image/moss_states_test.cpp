// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// MossText::read_states takes a states file only when its header's n x width rows are the rest of the file
// (Issue #83): a header claiming more rows than the file holds is refused before anything is allocated.  CPU only.
#include "strata/image/moss_text.hpp"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

namespace {

const char* const kPath = "moss_states_test.bin";

void write_states(int32_t n, int32_t width, size_t floats) {
    std::ofstream f(kPath, std::ios::binary | std::ios::trunc);
    const int32_t hdr[3] = {0x31455453, n, width};
    f.write(reinterpret_cast<const char*>(hdr), sizeof hdr);
    std::vector<float> rows(floats);
    for (size_t i = 0; i < floats; ++i) rows[i] = (float) i;
    f.write(reinterpret_cast<const char*>(rows.data()), (std::streamsize) (floats * sizeof(float)));
}

bool read(std::vector<float>& rows, int64_t& n, int64_t& width) {
    std::string err;
    return strata::image::MossText::read_states(kPath, rows, n, width, err);
}

}  // namespace

int main() {
    std::vector<float> rows;
    int64_t n = 0, width = 0;
    write_states(3, 4, 12);
    CHECK(read(rows, n, width) && n == 3 && width == 4 && rows.size() == 12 && rows[11] == 11.0f);
    std::vector<float> none;
    write_states(46341, 46341, 0);   // 12 bytes claiming 8.6 GiB of rows
    CHECK(!read(none, n, width) && none.capacity() == 0);
    write_states(3, 4, 11);          // one float short
    CHECK(!read(rows, n, width));
    write_states(3, 4, 13);          // one float over
    CHECK(!read(rows, n, width));
    write_states(-3, 4, 12);
    CHECK(!read(rows, n, width));
    std::remove(kPath);
    if (fails) return 1;
    std::printf("moss_states_test: ok\n");
    return 0;
}
