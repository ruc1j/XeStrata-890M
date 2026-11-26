// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/kernels/xe/q8_1_rows_item.hpp - one work-item of quantize_q8_1_rows (iq_kernels.cpp) for rows of whole 256s: the
// experts' activations as q8_1 blocks with canon_math.hpp's scale, one scale for each 256 values.  For the kernels
// that run it: quantize_q8_1_rows itself and the launch that builds the resident plan (verify_kernels.cpp), which
// writes the experts' image beside the plan.
#pragma once

#include "strata/kernels/canon_math.hpp"
#include "strata/kernels/q8_act.hpp"

#include <sycl/sycl.hpp>

#include <cstdint>

namespace strata::kernels {

// Work-item j of the launch: values 8j .. 8j + 7 of x; the sub-group (32 work-items) is one 256-value block and
// shares its scale.  n8: the values / 8; an item past it only takes part in the sub-group's reductions.  y: the q8_1
// blocks (36 bytes each: the fp16 scale and sum, then 32 int8 values).
inline void q8_1_rows_item(const sycl::sub_group& sg, long long j, long long n8, const float* x, uint8_t* y) {
    const bool in = j < n8;
    float v[8], a = 0.0f, sum = 0.0f;
    for (int q = 0; q < 8; ++q) {
        v[q] = in ? x[8 * j + q] : 0.0f;
        a = sycl::fmax(a, sycl::fabs(v[q]));
        sum += v[q];
    }
    const float amax = sycl::reduce_over_group(sg, a, sycl::maximum<float>());   // 32 lanes: 256 values
    sum += sycl::permute_group_by_xor(sg, sum, 1);
    sum += sycl::permute_group_by_xor(sg, sum, 2);
    if (!in) return;
    const float d = canon::q8_scale(amax), id = canon::q8_inv(d);
    const long long ib = j / 4, o = 8 * (j % 4);
    uint8_t* const block = y + ib * 36;
    for (int q = 0; q < 8; ++q) reinterpret_cast<int8_t*>(block + 4)[o + q] = canon::q8_value(v[q], id);
    if (o == 0) *reinterpret_cast<sycl::half2*>(block) = sycl::half2(sycl::half(d), sycl::half(q8act::finite(sum)));
}

}  // namespace strata::kernels
