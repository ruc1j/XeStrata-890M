// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/kernels/xe/router_top10_token.hpp - one token's top-10 by one sub-group, the body of router.cpp's native router
// (the pinned ggml topk-moe contract, see router.cpp), for the kernels that run it: the router's own launches and
// the launch that builds the resident plan behind it (verify_kernels.cpp).
//
// native_router follows ggml-cuda's topk-moe at llama.cpp 3cf03257; MIT License, Copyright (c) 2023-2026 The ggml
// authors, see third_party/main/ggml/LICENSE.
#pragma once

#include "xe_shuffle.hpp"

#include <sycl/sycl.hpp>

#include <cfloat>
#include <cmath>
#include <cstdint>

namespace strata::kernels::router_detail {

// The argmax order of both routers: the larger probability wins, a tie keeps the lower expert index.  It is a total
// order, so the xor butterflies below reach the same winner as CUDA's shuffle-down trees.
inline void better(float& bv, int& bi, float ov, int oi) {
    if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
}

// One token's top-10 by one sub-group (the pinned kernel's 32 x 8 block has only its first warp active), NPL = n_expert
// / 32 values a lane: `ids2` and `weights2`, when given, get the same values (the doorbell's mapped copies, written here
// instead of copied after)
template <int NPL>
inline void top10_token(const sycl::sub_group& sg, const float* logits, int32_t* ids, float* weights, int32_t* ids2,
                        float* weights2) {
        const int lane = (int) sg.get_local_linear_id();
        float values[NPL];
        for (int i = 0; i < NPL; ++i) values[i] = logits[lane + i * 32];
        float maximum = -INFINITY;
        for (int i = 0; i < NPL; ++i) maximum = sycl::max(maximum, values[i]);
        for (unsigned mask = 16; mask; mask >>= 1u) maximum = sycl::fmax(maximum, strata_shfl_xor(sg, maximum, mask));
        float sum = 0.0f;
        for (int i = 0; i < NPL; ++i) {
            values[i] = sycl::exp(values[i] - maximum);
            sum += values[i];
        }
        for (unsigned mask = 16; mask; mask >>= 1u) sum += strata_shfl_xor(sg, sum, mask);
        const float reciprocal = 1.0f / sum;
        for (int i = 0; i < NPL; ++i) {
            values[i] *= reciprocal;
            if (sycl::isnan(values[i])) values[i] = -FLT_MAX;
        }
        float selected = 0.0f, selected_sum = 0.0f;
        for (int rank = 0; rank < 10; ++rank) {
            float best = values[0];
            int expert = lane;
            for (int i = 1; i < NPL; ++i) {
                if (values[i] > best) { best = values[i]; expert = lane + i * 32; }
            }
            for (unsigned mask = 16; mask; mask >>= 1u)
                better(best, expert, strata_shfl_xor(sg, best, mask), strata_shfl_xor(sg, expert, mask));
            if (expert % 32 == lane) {   // (expert is never negative)
                values[expert / 32] = -INFINITY;
                ids[rank] = expert;
                if (ids2 != nullptr) ids2[rank] = expert;
                // Deliberately accumulate by WINNING EXPERT lane, not output rank.
                // Multiple selected experts in one lane add in selection order.
                selected_sum += best;
            }
            if (rank == lane) selected = best;
        }
        for (unsigned mask = 16; mask; mask >>= 1u) selected_sum += strata_shfl_xor(sg, selected_sum, mask);
        selected_sum = sycl::max(selected_sum, 6.103515625e-5f);
        const float inverse_selected_sum = 1.0f / selected_sum;
        if (lane < 10) {
            weights[lane] = selected * inverse_selected_sum;
            if (weights2 != nullptr) weights2[lane] = selected * inverse_selected_sum;
        }
}

}  // namespace strata::kernels::router_detail
