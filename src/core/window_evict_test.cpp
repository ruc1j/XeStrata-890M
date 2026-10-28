// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-License-Identifier: MIT
// window_evict_test: pick_window_victim (#3702) - the largest captured window size other than the one being captured
// and the prelaunched one; 0 when none is left.
#include "strata/core/window_evict.hpp"

#include <cstdio>

int main() {
    using strata::core::pick_window_victim;
    int failed = 0;
    auto check = [&](bool ok, const char* what) {
        if (!ok) { std::printf("FAIL: %s\n", what); ++failed; }
    };
    bool have[9] = {false, true, true, false, true, false, true, false, true};   // sizes 1, 2, 4, 6, 8
    check(pick_window_victim(have, 8, 3, 0) == 8, "the largest size goes first");
    check(pick_window_victim(have, 8, 8, 0) == 6, "not the size being captured");
    check(pick_window_victim(have, 8, 8, 6) == 4, "not the prelaunched size either");
    check(pick_window_victim(have, 4, 3, 0) == 4, "max_t bounds the search");
    bool none[9] = {};
    check(pick_window_victim(none, 8, 2, 0) == 0, "nothing captured: nothing to free");
    bool only[9] = {false, false, false, true};
    check(pick_window_victim(only, 8, 3, 0) == 0, "only the kept size: nothing to free");
    check(pick_window_victim(only, 8, 2, 3) == 0, "only the pinned size: nothing to free");
    if (failed == 0) std::printf("window_evict_test: OK\n");
    return failed == 0 ? 0 : 1;
}
