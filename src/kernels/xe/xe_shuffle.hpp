// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/xe_shuffle.hpp - a sub-group's xor exchange (sycl::permute_group_by_xor) on AMD's cross-lane paths.
//
// XeStrata-890M: intel/llvm's HIP backend lowers permute_group_by_xor to ds_bpermute_b32, a trip through the LDS
// crossbar of tens of cycles, and the kernels' reductions are chains of five of them (the hyper-connection read's down
// kernel ran 100 at T 4).  RDNA does the same exchanges in the vector ALU: xor 1 and 2 are DPP quad permutes, xor 4
// and 8 DPP row_xmask (gfx10 and later), xor 16 v_permlanex16_b32 with the identity selects.  Every lane reads the
// same lane's value as before, so every sum is bitwise the same.  Other offsets, other value sizes and other targets
// keep permute_group_by_xor.
#pragma once

#include <sycl/sycl.hpp>

#include <cstdint>
#include <type_traits>

template<class T>
inline T strata_shfl_xor(const sycl::sub_group& sg, T v, unsigned o) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__AMDGCN__) && (defined(__GFX10__) || defined(__GFX11__) || defined(__GFX12__))
    if constexpr (sizeof(T) == 4 && std::is_trivially_copyable_v<T>) {
        const int x = sycl::bit_cast<int>(v);
        int r;
        switch (o) {
            case 1: r = __builtin_amdgcn_mov_dpp(x, 0xB1, 0xF, 0xF, false); break;    // quad_perm [1,0,3,2]
            case 2: r = __builtin_amdgcn_mov_dpp(x, 0x4E, 0xF, 0xF, false); break;    // quad_perm [2,3,0,1]
            case 4: r = __builtin_amdgcn_mov_dpp(x, 0x164, 0xF, 0xF, false); break;   // row_xmask:4
            case 8: r = __builtin_amdgcn_mov_dpp(x, 0x168, 0xF, 0xF, false); break;   // row_xmask:8
            case 16: r = __builtin_amdgcn_permlanex16(x, x, 0x76543210u, 0xFEDCBA98u, false, false); break;
            default: return sycl::permute_group_by_xor(sg, v, o);
        }
        return sycl::bit_cast<T>(r);
    }
#endif
    return sycl::permute_group_by_xor(sg, v, o);
}
