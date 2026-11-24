// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// The SYCL Inkling-2-Mini kernels decode E2M1 (FP4) with integer ops instead of a table (a run-time indexed table can
// sit in private memory on a GPU).  The decode must equal the table bit for bit for every code, including the sign of
// zero.  No GPU.
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "sycl/inkling2mini_sy_e2m1.h"

static uint32_t bits(float f) {
    uint32_t b;
    std::memcpy(&b, &f, 4);
    return b;
}

int main() {
    int bad = 0;
    for (uint32_t nib = 0; nib < 256; ++nib) {   // also the high bits: only the low four count
        const float a = inkling2mini_sycl::e2m1_lut(nib), b = inkling2mini_sycl::e2m1_table(nib);
        if (bits(a) != bits(b)) {
            std::printf("FAIL code %u: lut %08x table %08x\n", nib, bits(a), bits(b));
            ++bad;
        }
    }
    const float expect[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    for (uint32_t m = 0; m < 8; ++m) {
        if (inkling2mini_sycl::e2m1_lut(m) != expect[m] || inkling2mini_sycl::e2m1_lut(m | 8u) != -expect[m]) {
            std::printf("FAIL magnitude %u\n", m);
            ++bad;
        }
    }
    if (bits(inkling2mini_sycl::e2m1_lut(8u)) != 0x80000000u) {
        std::printf("FAIL code 8 is not -0.0\n");
        ++bad;
    }
    if (bad) return 1;
    std::printf("inkling2mini_sy_e2m1_test: OK\n");
    return 0;
}
