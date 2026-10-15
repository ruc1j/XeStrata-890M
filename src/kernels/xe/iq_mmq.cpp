// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/kernels/xe/iq_mmq.cpp - see include/strata/kernels/iq_mmq.hpp.
//
// The weights' values and scales follow llama.cpp's dequantizers (ggml/src/ggml-quants.c, dequantize_row_*; MIT
// license, third_party/main/ggml/LICENSE, quoted in iq_kernels.cpp): a block of 32 values (16 for IQ2_XS, IQ2_S and
// IQ1_M) is its grid or table bytes, signed, times d and the block's scale (IQ1_M's values times 8, its scales over 8).
// Q8_0, Q5_0 and Q6_K (dequantize_row_q8_0, _q5_0, _q6_K) are their bytes, Q5_0's 5-bit values less 16 and Q6_K's
// 6-bit values less 32, times d and the scale of each 16.  Q4_K and Q5_K (dequantize_row_q4_K, _q5_K) are their 4- or
// 5-bit values times d sc, less dmin m: a block's product is d sc times the int8 sum, less dmin m times the
// activations' block sum, which the activations carry beside their scales (q8_1's s).  The activations are rounded as its q8_1 (amax / 127 a block of 32).
//
// A work-group takes a tile of one expert's rows (16, 32 or 64, by the group's largest) and 128 of its outputs, K 128
// values a step: the step's activation rows copied to local memory, the weight rows decoded there (a work-item a block
// of 32), then each sub-group's tiles multiplied a block at a time into int32 accumulators, which are scaled by the
// block's two scales into FP32 ones.  joint_matrix_apply gives a lane its elements but not where they sit in the tile,
// so the scales come as tiles too: loaded with stride 0, a column of one row's scale and a row of one output's, and
// multiplied element by element with the int32 tile (iq_mmq_usable checks that an int32 tile and an FP32 one hold
// their elements in the same order).  Intel's XMX take tiles of 8 x N x 32 on N lanes instead (intel_body), RDNA3 and
// RDNA4 their WMMA instructions.
//
// In a source file of its own, so its kernels are in a device image apart from the Intel-only ones.
#include "strata/kernels/iq_mmq.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/matrix_report.hpp"
#include "device_target.hpp"
#include "amd_wmma.hpp"
#include "strata/core/per_device.hpp"
#include "iq_bits.hpp"
#include "cuda_intrinsics.hpp"
#define STRATA_DEVICE_ARCH_TAG_TU xe_iq_mmq
#include "strata/kernels/device_arch_tag.hpp"

#define GGML_COMMON_DECL_SYCL
#define GGML_COMMON_IMPL_SYCL
#include "ggml-common.h"

#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <type_traits>

namespace strata::kernels {
namespace {

namespace mx = sycl::ext::oneapi::experimental::matrix;
namespace syclex = sycl::ext::oneapi::experimental;
using namespace iq_bits;
using u32x4 = uint32_t __attribute__((ext_vector_type(4)));   // one 16-byte access (sycl::uint4 is split on NVPTX)

// The work-group's tile shape (NVIDIA's int8 one on a warp, which RDNA's WMMA takes too; intel_body splits it into
// Intel's) and the K step.
constexpr int TM = 16, TN = 16, TK = 16, SG = 32;
[[maybe_unused]] constexpr int NE = TM * TN / SG;   // accumulator elements a lane (the layout check)
constexpr int KC = 128;               // K a step: four blocks of 32
constexpr int NB = KC / 32;
constexpr int LDA = KC + 16;          // local row strides in bytes (16-byte multiples, rows apart in the banks)
constexpr int LDB = KC + 16;

// AMD's device compile has joint_matrix on CDNA's matrix cores only, not for this shape: the joint_matrix bodies are
// left out (their types do not depend on a template parameter, so `if constexpr` alone does not drop them).  RDNA3 and
// RDNA4 (GFX11, GFX12) take the same products through their int8 WMMA instructions instead (STRATA_IQ_MMQ_WMMA).
#if defined(__SYCL_DEVICE_ONLY__) && defined(__AMDGCN__)
#define STRATA_IQ_MMQ_MATRIX 0
#if defined(__GFX11__) || defined(__GFX12__)
#define STRATA_IQ_MMQ_WMMA 1
#else
#define STRATA_IQ_MMQ_WMMA 0
#endif
#else
#define STRATA_IQ_MMQ_MATRIX 1
#define STRATA_IQ_MMQ_WMMA 0
#endif
// Intel's XMX (Xe2, Xe-HPG) take int8 tiles of 8 x N x 32 on N lanes, B in Intel's VNNI layout (the IS bodies): only
// in Intel's device compile, whose run-time check (intel_shape) is the only one that launches them
#if STRATA_IQ_MMQ_MATRIX && !STRATA_DEVICE_NOT_INTEL
#define STRATA_IQ_MMQ_INTEL 1
#else
#define STRATA_IQ_MMQ_INTEL 0
#endif

template<int S>
constexpr bool built() {
#if STRATA_IQ_MMQ_WMMA
    return true;
#elif !STRATA_IQ_MMQ_MATRIX
    return false;
#else
    return STRATA_NV_ARCH == 0 || STRATA_NV_ARCH >= 720;   // int8 on NVIDIA's tensor cores from sm_72 on
#endif
}

// ---------------------------------------------------------------- the weights' blocks of 32 values
// v[j]: values 4j .. 4j + 3 as signed bytes; s0, s1: the scales of values 0-15 and 16-31.
struct Sub {
    uint32_t v[8];
    float s0, s1;
    float m = 0.0f;   // the K-quants' min term: the block's values less m (Q4_K, Q5_K)
};
// the grid bytes (below 128) negated where m is 0xFF
inline uint32_t signed_bytes(uint32_t g, uint32_t m) { return (g ^ m) + (m & 0x01010101u); }

// The i-quant decoders the expert products meet most (IQ2_XXS, IQ2_XS, IQ2_S, IQ1_M, Q2_0) in two parts
// (XeStrata-890M): load gathers a block's bytes (Raw), decode makes the Sub from them, so that a lane's blocks of a
// step have their loads in flight together before any decode (the WMMA body); get is the two in sequence.  The
// unaligned words are read as the aligned words around them and joined by a shift, without get_int2_b2's branch (a
// word that would lie past the block is read from the block instead: no bytes past the row).  Measured on the 890M,
// IQ2_XS experts of 64 rows: each decode's loads were issued and waited for one after another (flat_load, s_waitcnt
// vmcnt(0), then the grid's).
struct Dw3 { uint32_t w0, w1, w2; };
inline Dw3 load_dw3(const void* p) {   // the aligned words around 8 bytes at a 2-byte aligned address
    const uintptr_t a = (uintptr_t) p;
    const uint32_t* w = (const uint32_t*) (a & ~(uintptr_t) 3);
    return {w[0], w[1], w[(a & 2) ? 2 : 1]};
}
inline U2 join8(const Dw3& w, const void* p) {   // those 8 bytes as two words
    return ((uintptr_t) p & 2) ? U2{(w.w0 >> 16) | (w.w1 << 16), (w.w1 >> 16) | (w.w2 << 16)} : U2{w.w0, w.w1};
}
struct Dw2 { uint32_t w0, w1; };
inline Dw2 load_dw2(const void* p) {   // the aligned words around 4 bytes at a 2-byte aligned address
    const uintptr_t a = (uintptr_t) p;
    const uint32_t* w = (const uint32_t*) (a & ~(uintptr_t) 3);
    return {w[0], w[(a & 2) ? 1 : 0]};
}
inline uint32_t join4(const Dw2& w, const void* p) { return ((uintptr_t) p & 2) ? (w.w0 >> 16) | (w.w1 << 16) : w.w0; }
template<typename T, typename = void> struct split_dec : std::false_type {};
template<typename T> struct split_dec<T, std::void_t<typename T::Raw>> : std::true_type {};

template<int TY> struct Dec;
template<> struct Dec<16> {   // IQ2_XXS: d (0.5 + ls) / 4
    static constexpr int QK = 256, BSZ = (int) sizeof(block_iq2_xxs);
    using Grid = uint64_t;
    static constexpr int GRID_BYTES = (int) sizeof(iq2xxs_grid);
    static const Grid* table() { return iq2xxs_grid; }
    struct Raw { Dw3 q; uint16_t d; };
    static Raw load(const uint8_t* row, int sb, size_t) {
        const block_iq2_xxs* b = reinterpret_cast<const block_iq2_xxs*>(row) + sb / 8;
        return {load_dw3(b->qs + 4 * (sb % 8)), sycl::bit_cast<uint16_t>(b->d)};
    }
    static Sub decode(const Raw& w, const uint8_t* row, int sb, size_t, const Grid* grid) {
        const block_iq2_xxs* b = reinterpret_cast<const block_iq2_xxs*>(row) + sb / 8;
        const U2 pk = join8(w.q, b->qs + 4 * (sb % 8));
        const uint32_t q2 = pk.x, aux = pk.y;
        Sub r;
        for (int k = 0; k < 8; k += 2) {
            const Int2 gp = grid_pair(grid, byte_of(q2, k / 2));
            const uint32_t s8 = ksigns_byte((uint8_t) (aux >> (7 * k / 2)));
            r.v[k] = signed_bytes((uint32_t) gp.x, sign_mask(s8));
            r.v[k + 1] = signed_bytes((uint32_t) gp.y, sign_mask(s8 >> 4));
        }
        r.s0 = r.s1 = (float) sycl::bit_cast<sycl::half>(w.d) * (float) (2 * (aux >> 28) + 1) * 0.125f;
        return r;
    }
    static Sub get(const uint8_t* row, int sb, size_t rb) { return decode(load(row, sb, rb), row, sb, rb, table()); }
};
template<> struct Dec<18> {   // IQ3_XXS: d (0.5 + ls) / 2
    static constexpr int QK = 256, BSZ = (int) sizeof(block_iq3_xxs);
    static Sub get(const uint8_t* row, int sb, [[maybe_unused]] size_t row_bytes) {
        const block_iq3_xxs* b = reinterpret_cast<const block_iq3_xxs*>(row) + sb / 8;
        const int ib = sb % 8;
        const U2 pk = get_int2_b2(b->qs, 2 * ib);
        const uint32_t packed[2] = {pk.x, pk.y};
        const uint32_t aux = (uint32_t) get_int_b2(b->qs, QK_K / 16 + ib);
        Sub r;
        for (int l = 0; l < 8; l += 2) {
            const uint32_t s8 = ksigns_byte((uint8_t) (aux >> (7 * l / 2)));
            r.v[l] = signed_bytes(iq3xxs_grid[byte_of(packed[l / 4], l % 4)], sign_mask(s8));
            r.v[l + 1] = signed_bytes(iq3xxs_grid[byte_of(packed[l / 4], (l + 1) % 4)], sign_mask(s8 >> 4));
        }
        r.s0 = r.s1 = (float) b->d * (float) (2 * (aux >> 28) + 1) * 0.25f;
        return r;
    }
};
template<> struct Dec<21> {   // IQ3_S: d (1 + 2 ls)
    static constexpr int QK = 256, BSZ = (int) sizeof(block_iq3_s);
    static Sub get(const uint8_t* row, int sb, [[maybe_unused]] size_t row_bytes) {
        const block_iq3_s* b = reinterpret_cast<const block_iq3_s*>(row) + sb / 8;
        const int ib = sb % 8;
        const U2 pk = get_int2_b2(b->qs, 2 * ib);
        const uint32_t packed[2] = {pk.x, pk.y};
        const int qh = b->qh[ib];
        const uint32_t signs = (uint32_t) get_int_b2(b->signs, ib);
        Sub r;
        for (int l = 0; l < 8; l += 2) {
            const uint32_t s8 = byte_of(signs, l / 2);
            r.v[l] = signed_bytes(iq3s_grid[byte_of(packed[l / 4], l % 4) | ((qh << (8 - l)) & 0x100)], sign_mask(s8));
            r.v[l + 1] = signed_bytes(iq3s_grid[byte_of(packed[l / 4], (l + 1) % 4) | ((qh << (7 - l)) & 0x100)],
                                      sign_mask(s8 >> 4));
        }
        r.s0 = r.s1 = (float) b->d * (float) (1 + 2 * ((b->scales[ib / 2] >> (4 * (ib & 1))) & 0x0F));
        return r;
    }
};
template<> struct Dec<17> {   // IQ2_XS: d (0.5 + ls) / 4, a scale per 16
    static constexpr int QK = 256, BSZ = (int) sizeof(block_iq2_xs);
    using Grid = uint64_t;
    static constexpr int GRID_BYTES = (int) sizeof(iq2xs_grid);
    static const Grid* table() { return iq2xs_grid; }
    struct Raw { Dw3 q; uint16_t d; uint8_t sc; };
    static Raw load(const uint8_t* row, int sb, size_t) {
        const block_iq2_xs* b = reinterpret_cast<const block_iq2_xs*>(row) + sb / 8;
        const int ib = sb % 8;
        return {load_dw3(b->qs + 4 * ib), sycl::bit_cast<uint16_t>(b->d), b->scales[ib]};
    }
    static Sub decode(const Raw& w, const uint8_t* row, int sb, size_t, const Grid* grid) {
        const block_iq2_xs* b = reinterpret_cast<const block_iq2_xs*>(row) + sb / 8;
        const U2 pk = join8(w.q, b->qs + 4 * (sb % 8));
        const uint32_t packed[2] = {pk.x, pk.y};
        Sub r;
        for (int l = 0; l < 8; l += 2) {
            const uint16_t q2 = half_of(packed[l / 4], (l / 2) % 2);
            const Int2 gp = grid_pair(grid, q2 & 0x1FF);
            const uint32_t s8 = ksigns_byte((uint8_t) (q2 >> 9));
            r.v[l] = signed_bytes((uint32_t) gp.x, sign_mask(s8));
            r.v[l + 1] = signed_bytes((uint32_t) gp.y, sign_mask(s8 >> 4));
        }
        const float d = (float) sycl::bit_cast<sycl::half>(w.d);
        r.s0 = d * (float) (2 * (w.sc & 0x0F) + 1) * 0.125f;
        r.s1 = d * (float) (2 * (w.sc >> 4) + 1) * 0.125f;
        return r;
    }
    static Sub get(const uint8_t* row, int sb, size_t rb) { return decode(load(row, sb, rb), row, sb, rb, table()); }
};
template<> struct Dec<22> {   // IQ2_S: d (0.5 + ls) / 4, a scale per 16
    static constexpr int QK = 256, BSZ = (int) sizeof(block_iq2_s);
    using Grid = uint64_t;
    static constexpr int GRID_BYTES = (int) sizeof(iq2s_grid);
    static const Grid* table() { return iq2s_grid; }
    struct Raw { Dw2 q, sg; uint16_t d; uint8_t qh, sc; };
    static Raw load(const uint8_t* row, int sb, size_t) {
        const block_iq2_s* b = reinterpret_cast<const block_iq2_s*>(row) + sb / 8;
        const int ib = sb % 8;
        return {load_dw2(b->qs + 4 * ib), load_dw2(b->qs + QK_K / 8 + 4 * ib), sycl::bit_cast<uint16_t>(b->d), b->qh[ib],
                b->scales[ib]};
    }
    static Sub decode(const Raw& w, const uint8_t* row, int sb, size_t, const Grid* grid) {
        const block_iq2_s* b = reinterpret_cast<const block_iq2_s*>(row) + sb / 8;
        const int ib = sb % 8;
        const uint32_t qs = join4(w.q, b->qs + 4 * ib), signs = join4(w.sg, b->qs + QK_K / 8 + 4 * ib);
        const int qh = w.qh;
        Sub r;
        for (int l = 0; l < 8; l += 2) {
            const Int2 gp = grid_pair(grid, byte_of(qs, l / 2) | ((qh << (8 - l)) & 0x300));
            const uint32_t s8 = byte_of(signs, l / 2);
            r.v[l] = signed_bytes((uint32_t) gp.x, sign_mask(s8));
            r.v[l + 1] = signed_bytes((uint32_t) gp.y, sign_mask(s8 >> 4));
        }
        const float d = (float) sycl::bit_cast<sycl::half>(w.d);
        r.s0 = d * (float) (2 * (w.sc & 0x0F) + 1) * 0.125f;
        r.s1 = d * (float) (2 * (w.sc >> 4) + 1) * 0.125f;
        return r;
    }
    static Sub get(const uint8_t* row, int sb, size_t rb) { return decode(load(row, sb, rb), row, sb, rb, table()); }
};
template<> struct Dec<20> {   // IQ4_NL: d
    static constexpr int QK = 32, BSZ = (int) sizeof(block_iq4_nl);
    static Sub get(const uint8_t* row, int sb, [[maybe_unused]] size_t row_bytes) {
        const block_iq4_nl* b = reinterpret_cast<const block_iq4_nl*>(row) + sb;
        Sub r;
        for (int j = 0; j < 4; ++j) {
            const Int2 t = get_int_from_table_16(get_int_b2(b->qs, j), kvalues_iq4nl);
            r.v[j] = (uint32_t) t.x;
            r.v[4 + j] = (uint32_t) t.y;
        }
        r.s0 = r.s1 = (float) b->d;
        return r;
    }
};
template<> struct Dec<23> {   // IQ4_XS: d (ls - 32)
    static constexpr int QK = 256, BSZ = (int) sizeof(block_iq4_xs);
    static Sub get(const uint8_t* row, int sb, [[maybe_unused]] size_t row_bytes) {
        const block_iq4_xs* b = reinterpret_cast<const block_iq4_xs*>(row) + sb / 8;
        const int ib = sb % 8;
        Sub r;
        for (int j = 0; j < 4; ++j) {
            const Int2 t = get_int_from_table_16(get_int_b4(b->qs, 4 * ib + j), kvalues_iq4nl);
            r.v[j] = (uint32_t) t.x;
            r.v[4 + j] = (uint32_t) t.y;
        }
        const int ls = (((b->scales_l[ib / 2] >> (4 * (ib & 1))) & 0x0F) | (((b->scales_h >> (2 * ib)) & 0x03) << 4)) - 32;
        r.s0 = r.s1 = (float) b->d * (float) ls;
        return r;
    }
};
template<> struct Dec<42> {   // Q2_0: d (code - 1), 64 values a block
    static constexpr int QK = 64, BSZ = (int) sizeof(block_q2_0);
    using Grid = uint32_t;
    static constexpr int GRID_BYTES = 0;
    static const Grid* table() { return nullptr; }
    struct Raw { Dw3 q; uint16_t d; };
    static Raw load(const uint8_t* row, int sb, size_t) {
        const block_q2_0* b = reinterpret_cast<const block_q2_0*>(row) + sb / 2;
        return {load_dw3(b->qs + 8 * (sb % 2)), sycl::bit_cast<uint16_t>(b->d)};
    }
    static Sub decode(const Raw& w, const uint8_t* row, int sb, size_t, const Grid*) {
        const block_q2_0* b = reinterpret_cast<const block_q2_0*>(row) + sb / 2;
        const U2 pk = join8(w.q, b->qs + 8 * (sb % 2));
        const uint32_t packed[2] = {pk.x, pk.y};
        Sub r;
        for (int j = 0; j < 4; ++j) {
            const uint32_t q = half_of(packed[j / 2], j % 2);
            const int k = 2 * j;
            r.v[k] = q2_0_bytes(q & 0xFFu);
            r.v[k + 1] = q2_0_bytes(q >> 8);
        }
        r.s0 = r.s1 = (float) sycl::bit_cast<sycl::half>(w.d);
        return r;
    }
    static Sub get(const uint8_t* row, int sb, size_t rb) { return decode(load(row, sb, rb), row, sb, rb, nullptr); }
};

template<> struct Dec<29> {   // IQ1_M: d (2 ls + 1) (grid +- 1/8), a scale per 16
    static constexpr int QK = 256, BSZ = (int) sizeof(block_iq1_m);
    using Grid = uint32_t;
    static constexpr int GRID_BYTES = (int) sizeof(iq1s_grid_gpu);
    static const Grid* table() { return iq1s_grid_gpu; }
    struct Raw { Dw2 q; Dw3 sc; uint16_t qh; };
    static Raw load(const uint8_t* row, int sb, size_t) {
        const block_iq1_m* b = reinterpret_cast<const block_iq1_m*>(row) + sb / 8;
        const int ib = sb % 8;
        return {load_dw2(b->qs + 4 * ib), load_dw3(b->scales),
                (uint16_t) (b->qh[2 * ib] | (b->qh[2 * ib + 1] << 8))};
    }
    static Sub decode(const Raw& w, const uint8_t* row, int sb, size_t, const Grid* grid) {
        const block_iq1_m* b = reinterpret_cast<const block_iq1_m*>(row) + sb / 8;
        const int ib = sb % 8;
        const uint32_t qs = join4(w.q, b->qs + 4 * ib);
        const U2 scw = join8(w.sc, b->scales);
        const uint16_t sc[4] = {half_of(scw.x, 0), half_of(scw.x, 1), half_of(scw.y, 0), half_of(scw.y, 1)};
        Sub r;
        for (int k = 0; k < 8; k += 2) {
            const int l = k / 2;
            const int qh = w.qh >> (4 * l);
            const uint32_t g = grid[byte_of(qs, l) | ((qh & 0x07) << 8)];
            // the grid's nibbles (0 .. 2, the value + 1) as 8 (value +- 1/8): 8 nibble + 9 or 7, then 16 off each
            // byte without a borrow crossing bytes (as q2_0_bytes)
            const uint32_t c = qh & 0x08 ? 0x07070707u : 0x09090909u;
            r.v[k] = (((((g & 0x0F0F0F0Fu) << 3) + c) | 0x80808080u) - 0x10101010u) ^ 0x80808080u;
            r.v[k + 1] = ((((((g >> 4) & 0x0F0F0F0Fu) << 3) + c) | 0x80808080u) - 0x10101010u) ^ 0x80808080u;
        }
        const float d = iq1m_scale(sc) * 0.125f;
        const int ls = sc[ib / 2] >> (6 * (ib % 2));
        r.s0 = d * (float) (2 * (ls & 0x07) + 1);
        r.s1 = d * (float) (2 * ((ls >> 3) & 0x07) + 1);
        return r;
    }
    static Sub get(const uint8_t* row, int sb, size_t rb) { return decode(load(row, sb, rb), row, sb, rb, table()); }
};

template<> struct Dec<8> {   // Q8_0: d
    static constexpr int QK = 32, BSZ = (int) sizeof(block_q8_0);
    static Sub get(const uint8_t* row, int sb, [[maybe_unused]] size_t row_bytes) {
        const block_q8_0* b = reinterpret_cast<const block_q8_0*>(row) + sb;
        Sub r;
        for (int j = 0; j < 8; ++j) r.v[j] = (uint32_t) get_int_b2(b->qs, j);
        r.s0 = r.s1 = (float) b->d;
        return r;
    }
};
template<> struct Dec<14> {   // Q6_K: d scale (6-bit value - 32), a scale per 16
    static constexpr int QK = 256, BSZ = (int) sizeof(block_q6_K);
    static Sub get(const uint8_t* row, int sb, [[maybe_unused]] size_t row_bytes) {
        const block_q6_K* b = reinterpret_cast<const block_q6_K*>(row) + sb / 8;
        // block of 32 ib: half n of the 256 (ql, qh and the scales 64, 32 and 8 bytes on), quarter j of the half: the
        // low nibbles (j 0, 1) or high ones (j 2, 3) of ql's first or second 32 bytes, and qh's 2 bits at 2 j
        const int ib = sb % 8, n = ib / 4, j = ib % 4;
        Sub r;
        for (int k = 0; k < 8; ++k) {
            const uint32_t ql = (uint32_t) get_int_b2(b->ql, 16 * n + 8 * (j & 1) + k);
            const uint32_t qh = (uint32_t) get_int_b2(b->qh, 8 * n + k);
            const uint32_t v = ((ql >> (4 * (j >> 1))) & 0x0F0F0F0Fu) | (((qh >> (2 * j)) & 0x03030303u) << 4);
            r.v[k] = ((v | 0x80808080u) - 0x20202020u) ^ 0x80808080u;   // less 32 in each byte, no borrow across
        }
        const float d = (float) b->d;
        r.s0 = d * (float) b->scales[8 * n + 2 * j];
        r.s1 = d * (float) b->scales[8 * n + 2 * j + 1];
        return r;
    }
};

template<> struct Dec<6> {   // Q5_0: d ((5-bit value) - 16), the high bits in qh
    static constexpr int QK = 32, BSZ = (int) sizeof(block_q5_0);
    static Sub get(const uint8_t* row, int sb, [[maybe_unused]] size_t row_bytes) {
        const block_q5_0* b = reinterpret_cast<const block_q5_0*>(row) + sb;
        const uint32_t qh = (uint32_t) get_int_b2(b->qh, 0);
        Sub r;
        for (int k = 0; k < 4; ++k) {   // values 4k .. 4k + 3 in the low nibbles, 16 + 4k .. in the high ones
            const uint32_t ql = (uint32_t) get_int_b2(b->qs, k);
            uint32_t h0 = 0, h1 = 0;
            for (int t = 0; t < 4; ++t) {
                h0 |= ((qh >> (4 * k + t)) & 1u) << (8 * t + 4);
                h1 |= ((qh >> (4 * k + t + 16)) & 1u) << (8 * t + 4);
            }
            r.v[k] = ((((ql & 0x0F0F0F0Fu) | h0) | 0x80808080u) - 0x10101010u) ^ 0x80808080u;
            r.v[4 + k] = (((((ql >> 4) & 0x0F0F0F0Fu) | h1) | 0x80808080u) - 0x10101010u) ^ 0x80808080u;
        }
        r.s0 = r.s1 = (float) b->d;
        return r;
    }
};
// Q4_K and Q5_K's 6-bit scale and min of block j (get_scale_min_k4)
inline void scale_min_k4(int j, const uint8_t* q, int& d, int& m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}
template<> struct Dec<12> {   // Q4_K: d sc value - dmin m, block ib the low (even ib) or high nibbles of qs[32 (ib / 2)..]
    static constexpr int QK = 256, BSZ = (int) sizeof(block_q4_K);
    static Sub get(const uint8_t* row, int sb, [[maybe_unused]] size_t row_bytes) {
        const block_q4_K* b = reinterpret_cast<const block_q4_K*>(row) + sb / 8;
        const int ib = sb % 8;
        Sub r;
        for (int k = 0; k < 8; ++k)
            r.v[k] = ((uint32_t) get_int_b4(b->qs, 8 * (ib / 2) + k) >> (4 * (ib & 1))) & 0x0F0F0F0Fu;
        int sc, mn;
        scale_min_k4(ib, b->scales, sc, mn);
        r.s0 = r.s1 = (float) b->dm[0] * (float) sc;
        r.m = (float) b->dm[1] * (float) mn;
        return r;
    }
};
template<> struct Dec<13> {   // Q5_K: Q4_K's nibbles with bit ib of qh as the fifth
    static constexpr int QK = 256, BSZ = (int) sizeof(block_q5_K);
    static Sub get(const uint8_t* row, int sb, [[maybe_unused]] size_t row_bytes) {
        const block_q5_K* b = reinterpret_cast<const block_q5_K*>(row) + sb / 8;
        const int ib = sb % 8;
        Sub r;
        for (int k = 0; k < 8; ++k) {
            const uint32_t ql = (uint32_t) get_int_b4(b->qs, 8 * (ib / 2) + k) >> (4 * (ib & 1));
            const uint32_t qh = (uint32_t) get_int_b4(b->qh, k) >> ib;
            r.v[k] = (ql & 0x0F0F0F0Fu) | ((qh & 0x01010101u) << 4);
        }
        int sc, mn;
        scale_min_k4(ib, b->scales, sc, mn);
        r.s0 = r.s1 = (float) b->dm[0] * (float) sc;
        r.m = (float) b->dm[1] * (float) mn;
        return r;
    }
};

template<> struct Dec<11> {   // Q3_K: d (6-bit scale - 32) ((2-bit value | high bit << 2) - 4), a scale per 16
    static constexpr int QK = 256, BSZ = (int) sizeof(block_q3_K);
    // scale `is` of 16 (dequantize_row_q3_K's aux unpacking): its low nibble from bytes 0-7, its high bits from 8-11
    static int scale(const uint8_t* s, int is) {
        const int lo = is < 8 ? s[is] & 0x0F : s[is - 8] >> 4;
        const int hi = (s[8 + (is & 3)] >> (2 * (is >> 2))) & 0x03;
        return (lo | (hi << 4)) - 32;
    }
    static Sub get(const uint8_t* row, int sb, [[maybe_unused]] size_t row_bytes) {
        const block_q3_K* b = reinterpret_cast<const block_q3_K*>(row) + sb / 8;
        const int ib = sb % 8, n = ib / 4, j = ib % 4;   // 128-value half n, its 2-bit field j
        Sub r;
        for (int k = 0; k < 8; ++k) {   // values 0-15 from qs[32n..] and hmask[0..], 16-31 from qs[32n + 16..]
            const int l = k < 4 ? 4 * k : 16 + 4 * (k - 4);
            const uint32_t q = ((uint32_t) get_int_b2(b->qs + 32 * n, l / 4) >> (2 * j)) & 0x03030303u;
            const uint32_t h = ((uint32_t) get_int_b2(b->hmask, l / 4) >> (4 * n + j)) & 0x01010101u;
            r.v[k] = (((q | (h << 2)) | 0x80808080u) - 0x04040404u) ^ 0x80808080u;
        }
        const float d = (float) b->d;
        r.s0 = d * (float) scale(b->scales, 8 * n + 2 * j);
        r.s1 = d * (float) scale(b->scales, 8 * n + 2 * j + 1);
        return r;
    }
};
// Q2_K: d sc value - dmin m, sc and m a pair per 16 values.  The products have one min a block of 32, so a Q2_K row
// is read as twice its length: block 2 i is its sub-block i's values (scales d sc), block 2 i + 1 ones scaled by
// -dmin m, against activations whose blocks come twice each (iq_mmq_quantize's dup).  QK: 256 values per 512 read.
template<> struct Dec<10> {
    static constexpr int QK = 512, BSZ = (int) sizeof(block_q2_K);
    static Sub get(const uint8_t* row, int sb, [[maybe_unused]] size_t row_bytes) {
        const int rb = sb / 2;
        const block_q2_K* b = reinterpret_cast<const block_q2_K*>(row) + rb / 8;
        const int ib = rb % 8, n = ib / 4, j = ib % 4;
        const int sc0 = b->scales[8 * n + 2 * j], sc1 = b->scales[8 * n + 2 * j + 1];
        Sub r;
        if (sb % 2 == 0) {
            for (int k = 0; k < 8; ++k) {
                const int l = k < 4 ? 4 * k : 16 + 4 * (k - 4);
                r.v[k] = ((uint32_t) get_int_b4(b->qs + 32 * n, l / 4) >> (2 * j)) & 0x03030303u;
            }
            const float d = (float) b->dm[0];
            r.s0 = d * (float) (sc0 & 0x0F);
            r.s1 = d * (float) (sc1 & 0x0F);
        } else {
            for (int k = 0; k < 8; ++k) r.v[k] = 0x01010101u;
            const float dmin = (float) b->dm[1];
            r.s0 = -dmin * (float) (sc0 >> 4);
            r.s1 = -dmin * (float) (sc1 >> 4);
        }
        return r;
    }
};

template<> struct Dec<kNativeQ6KRows> {   // Q6_K split by row (native_mmvq.hpp): a row's ql, then qh, scales and d
    static constexpr int QK = 256, BSZ = (int) sizeof(block_q6_K);
    static Sub get(const uint8_t* row, int sb, size_t row_bytes) {
        const size_t bpr = row_bytes / BSZ, blk = (size_t) sb / 8;
        const int ib = sb % 8, n = ib / 4, j = ib % 4;
        const uint8_t* ql = row + blk * 128 + (size_t) 64 * n;
        const uint8_t* qh = row + bpr * 128 + blk * 64 + (size_t) 32 * n;
        const int8_t* sc = reinterpret_cast<const int8_t*>(row + bpr * 192 + blk * 16) + (size_t) 8 * n;
        Sub r;
        for (int k = 0; k < 8; ++k) {
            const uint32_t l = (uint32_t) get_int_b2(ql + 32 * (ptrdiff_t) (j & 1), k), h = (uint32_t) get_int_b2(qh, k);
            const uint32_t v = ((l >> (4 * (j >> 1))) & 0x0F0F0F0Fu) | (((h >> (2 * j)) & 0x03030303u) << 4);
            r.v[k] = ((v | 0x80808080u) - 0x20202020u) ^ 0x80808080u;
        }
        const float d = (float) *reinterpret_cast<const sycl::half*>(row + bpr * 208 + blk * 2);
        r.s0 = d * (float) sc[2 * (ptrdiff_t) j];
        r.s1 = d * (float) sc[2 * j + 1];
        return r;
    }
};

template<> struct Dec<kNativeQ8Rows> {   // Q8_0 split by row (native_mmvq.hpp): a row's qs, then its d
    static constexpr int QK = 32, BSZ = (int) sizeof(block_q8_0);
    static Sub get(const uint8_t* row, int sb, size_t row_bytes) {
        const size_t bpr = row_bytes / BSZ;
        const uint8_t* qs = row + (size_t) sb * 32;
        Sub r;
        for (int j = 0; j < 8; ++j) r.v[j] = (uint32_t) get_int_b4(qs, j);
        r.s0 = r.s1 = (float) *reinterpret_cast<const sycl::half*>(row + bpr * 32 + (size_t) sb * 2);
        return r;
    }
};

template<int TY>
constexpr bool half_scales() { return TY == 17 || TY == 22 || TY == 29 || TY == 14 || TY == kNativeQ6KRows || TY == 10 || TY == 11; }
template<int TY>
constexpr bool has_min() { return TY == 12 || TY == 13; }
template<int TY, bool = split_dec<Dec<TY>>::value> struct GridOf { using type = uint32_t; };
template<int TY> struct GridOf<TY, true> { using type = typename Dec<TY>::Grid; };
template<int TY>
constexpr int grid_bytes() {
    if constexpr (split_dec<Dec<TY>>::value) return Dec<TY>::GRID_BYTES;
    else return 0;
}

// ---------------------------------------------------------------- the products
template<typename T>
inline auto local_as(sycl::multi_ptr<uint8_t, sycl::access::address_space::local_space, sycl::access::decorated::no> p) {
    using V = sycl::multi_ptr<void, sycl::access::address_space::local_space, sycl::access::decorated::no>;
    return static_cast<sycl::multi_ptr<T, sycl::access::address_space::local_space, sycl::access::decorated::no>>(V(p));
}
constexpr int up128(int b) { return (b + 127) / 128 * 128; }

// An int32 sum below 2^22 in magnitude (a block's: 32 products of int8 values) as a float, exactly: its bits added to
// those of 1.5 * 2^23, less that.  Two full-rate instructions instead of a conversion (a quarter of the rate on the
// RTX 4070, where it was a top stall).
[[maybe_unused]] inline float exact_float(int32_t c) { return sycl::bit_cast<float>(c + 0x4B400000) - 12582912.0f; }

// SGM x 2 tiles a sub-group, NSGM x 4 sub-groups a work-group: WM rows by 128 outputs.  Local memory, one buffer
// carved at 128-byte boundaries (the matrix loads take 32-byte aligned tiles): the step's activations (WM x KC int8,
// row stride LDA) and weights (WN x KC, row stride LDB: B column-major), the activations' scales as [block][row] and
// the weights' as [block][half][output] (so that a tile of either loads with stride 0: the same scale down a column or
// along a row), and a tile a sub-group on its way out.  DP4A: the same tiles and steps without matrix engines (the
// GPUs iq_mmq_usable says no for: RDNA2, Volta, Intel's without XMX), each work-item a column of outputs down
// WM / (lanes / WN) of the rows, the products by dp4a.  IS: Intel's XMX shapes (1: Xe2's 8 x 16 x 32 on 16 lanes, 2:
// Xe-HPG's 8 x 8 x 32 on 8), the same work-group and steps (intel_body); a type with a scale per 16 values keeps two
// copies of the activations, each with the other half of every block zeroed.
template<int TY, int SGM, int NSGM, bool DP4A = false, bool TAIL = false, int IS = 0>
struct Kernel {
    static constexpr int SGN = 2, NSGN = 4, NSG = NSGM * NSGN;
    static constexpr int WM = NSGM * SGM * TM, WN = NSGN * SGN * TN;
    static constexpr bool HALF = half_scales<TY>();
    static constexpr bool MIN = has_min<TY>();
    static constexpr int ISG = IS == 1 ? 16 : IS == 2 ? 8 : SG;   // the sub-group size, and Intel's tile width
    static constexpr int A_N = IS != 0 && HALF ? 2 : 1;           // copies of the activations' tile
    // MIN: the activations' block sums as [block][row] and the weights' min terms as [block][output] after the rest
    // The output staging tile (ST) last: the WMMA body writes its accumulators straight out and does without it
    // (LOCAL_DIRECT, launch), so its work-groups fit a 64 KB local memory two at a time (the 890M, IQ2_XS: 40 -> 32 KB;
    // a K step of 64, 18 KB and three work-groups, was slower: 16 experts of 64 rows 1.86 -> 2.05 ms; the grid table
    // after it, 36 KB and one work-group for the 64-row tile, was faster still: 1.65 -> 1.61 ms, 16 rows 0.71 -> 0.65)
    static constexpr int A_OFF = 0, B_OFF = up128(A_OFF + A_N * WM * LDA), XS_OFF = up128(B_OFF + WN * LDB),
                         WS_OFF = up128(XS_OFF + NB * WM * 4), XU_OFF = up128(WS_OFF + 2 * NB * WN * 4),
                         WM_OFF = up128(XU_OFF + NB * WM * 4), ST_OFF = MIN ? up128(WM_OFF + NB * WN * 4) : XU_OFF,
                         LOCAL_DIRECT = ST_OFF, LOCAL = up128(ST_OFF + NSG * TM * TN * 4);
    // RDNA's body (LOCAL_DIRECT) keeps the type's grid table in local memory after the rest (XeStrata-890M): the
    // decode's second loads (the grid's pairs, behind the block's bytes) from there instead of global memory
    static constexpr int GRID_BYTES = grid_bytes<TY>();
    static constexpr int GRID_OFF = LOCAL_DIRECT, LOCAL_DIRECT_GRID = up128(GRID_OFF + GRID_BYTES);
    const uint8_t* w;
    const uint8_t* const* wtab;   // expert e's weights at wtab[e] + woff when set, else at w + e expert_bytes
    size_t woff, expert_bytes, row_bytes;
    const int8_t* xq;
    const float* xs;
    const float* xu;   // the activations' block sums (d times the int8 values' sum), MIN
    int64_t K;
    const int32_t* bounds;
    const int32_t* ids;
    float* dst;
    int64_t ld_dst, tiles_m, tiles_n;
    sycl::local_accessor<uint8_t, 1> lm;

    void dp4a_body(sycl::nd_item<1> it) const {
        constexpr int lanes = NSG * SG, RSTEP = lanes / WN, RPT = WM / RSTEP;
        const int64_t g = (int64_t) it.get_group(0), per = tiles_m * tiles_n, e = g / per, r = g % per;
        const int64_t m0 = r % tiles_m * WM, n0 = r / tiles_m * WN;
        const int64_t row0 = bounds[e], rows = bounds[e + 1] - row0;
        if (m0 >= rows) return;   // the whole work-group: this expert has fewer rows than the launch
        const int lid = (int) it.get_local_id(0), n = lid % WN, rbase = lid / WN;
        const uint8_t* wx = wtab != nullptr ? wtab[e] + woff : w + (size_t) e * expert_bytes;
        const auto base = lm.template get_multi_ptr<sycl::access::decorated::no>();
        int8_t* al = local_as<int8_t>(base + A_OFF).get();
        int8_t* bl = local_as<int8_t>(base + B_OFF).get();
        float* xsl = local_as<float>(base + XS_OFF).get();
        [[maybe_unused]] float* xul = local_as<float>(base + XU_OFF).get();
        [[maybe_unused]] float* wml = local_as<float>(base + WM_OFF).get();
        float* wsl = local_as<float>(base + WS_OFF).get();
        float acc[RPT];
        for (int i = 0; i < RPT; ++i) acc[i] = 0.0f;
        const int64_t kb = K / 32;   // activation scales a row
        for (int64_t k0 = 0; k0 < K; k0 += KC) {
            sycl::group_barrier(it.get_group());   // the last step's tiles are read
            const bool last = TAIL && k0 + KC > K;   // TAIL: a last step past K, zeros past it
            for (int v = lid; v < WM * (KC / 16); v += lanes) {
                const int rr = v / (KC / 16), c = v % (KC / 16) * 16;
                const int64_t row = m0 + rr;
                u32x4 val = {0u, 0u, 0u, 0u};
                if (row < rows && (!last || k0 + c < K)) val = *reinterpret_cast<const u32x4*>(xq + (row0 + row) * K + k0 + c);
                *reinterpret_cast<u32x4*>(al + static_cast<ptrdiff_t>(rr) * LDA + c) = val;
            }
            for (int v = lid; v < WM * NB; v += lanes) {
                const int b = v / WM, rr = v % WM;
                const int64_t row = m0 + rr;
                xsl[v] = row < rows && (!last || k0 / 32 + b < kb) ? xs[(row0 + row) * kb + k0 / 32 + b] : 0.0f;
                if constexpr (MIN) xul[v] = row < rows && (!last || k0 / 32 + b < kb) ? xu[(row0 + row) * kb + k0 / 32 + b] : 0.0f;
            }
            for (int v = lid; v < WN * NB; v += lanes) {
                const int nn = v / NB, b = v % NB;
                const Sub d = !last || k0 / 32 + b < kb ? Dec<TY>::get(wx + (size_t) (n0 + nn) * row_bytes, (int) (k0 / 32) + b, row_bytes)
                                                : Sub{{}, 0.0f, 0.0f, 0.0f};   // past K: zeros
                u32x4* o = reinterpret_cast<u32x4*>(bl + static_cast<ptrdiff_t>(nn) * LDB + static_cast<ptrdiff_t>(b) * 32);
                o[0] = u32x4{d.v[0], d.v[1], d.v[2], d.v[3]};
                o[1] = u32x4{d.v[4], d.v[5], d.v[6], d.v[7]};
                wsl[(2 * b) * WN + nn] = d.s0;
                wsl[(2 * b + 1) * WN + nn] = d.s1;
                if constexpr (MIN) wml[b * WN + nn] = d.m;
            }
            sycl::group_barrier(it.get_group());
            for (int b = 0; b < NB; ++b) {
                int wv[8];   // this work-item's output's block, read once for its rows
                const int* wp = reinterpret_cast<const int*>(bl + static_cast<ptrdiff_t>(n) * LDB + static_cast<ptrdiff_t>(b) * 32);
                for (int i = 0; i < 8; ++i) wv[i] = wp[i];
                const float ws0 = wsl[(2 * b) * WN + n], ws1 = wsl[(2 * b + 1) * WN + n];
                for (int i = 0; i < RPT; ++i) {
                    const int rr = rbase + i * RSTEP;
                    const int* ap = reinterpret_cast<const int*>(al + static_cast<ptrdiff_t>(rr) * LDA + static_cast<ptrdiff_t>(b) * 32);
                    int s0 = 0, s1 = 0;
                    for (int j = 0; j < 4; ++j) s0 = xe::dp4a(ap[j], wv[j], s0);
                    for (int j = 4; j < 8; ++j) s1 = xe::dp4a(ap[j], wv[j], s1);
                    acc[i] += xsl[b * WM + rr] * (ws0 * (float) s0 + ws1 * (float) s1);
                    if constexpr (MIN) acc[i] -= wml[b * WN + n] * xul[b * WM + rr];
                }
            }
        }
        for (int i = 0; i < RPT; ++i) {
            const int64_t row = m0 + rbase + (int64_t) i * RSTEP;
            if (row < rows) dst[(int64_t) ids[row0 + row] * ld_dst + n0 + n] = acc[i];
        }
    }

    // Intel's XMX: the sub-groups split the WM x WN tile into 8 x ISG tiles, a column of them each (TPS tiles down its
    // row group's rows).  The weights' words go to local memory in Intel's VNNI order, [K / 4][output] (a word is 4 K
    // values of one output, as Dec gives them); a block of 32 is one product.  An 8 x N accumulator gives each lane one
    // column, its element i in row i (Xe's DPAS layout, as qsa_prompt_attn takes it; intel_layout_ok checks it at the
    // start), so the scales are applied by element order: asking each element its coordinates cost more than the
    // products there.
    void intel_body(sycl::nd_item<1> it) const {
#if STRATA_IQ_MMQ_INTEL
        constexpr int IM = 8, IK = 32, IN = ISG;
        constexpr int NT = WN / IN, NSUB = NSG * SG / ISG, RG = NSUB / NT, TPS = WM / IM / RG;
        static_assert(NSUB % NT == 0 && (WM / IM) % RG == 0, "the tiles split evenly over the sub-groups");
        constexpr int LDBV = WN * 4;   // a VNNI row: 4 K values of every output
        const int64_t g = (int64_t) it.get_group(0), per = tiles_m * tiles_n, e = g / per, r = g % per;
        const int64_t m0 = r % tiles_m * WM, n0 = r / tiles_m * WN;
        const int64_t row0 = bounds[e], rows = bounds[e + 1] - row0;
        if (m0 >= rows) return;   // the whole work-group: this expert has fewer rows than the launch
        const auto sg = it.get_sub_group();
        const int sgid = (int) sg.get_group_linear_id(), lid = (int) it.get_local_id(0);
        const int lane = (int) sg.get_local_linear_id();
        constexpr int lanes = NSG * SG;
        const int rs = sgid / NT * TPS * IM, cs = sgid % NT * IN;   // the sub-group's first row and column in the tile
        const uint8_t* wx = wtab != nullptr ? wtab[e] + woff : w + (size_t) e * expert_bytes;
        const auto base = lm.template get_multi_ptr<sycl::access::decorated::no>();
        const auto ap = local_as<int8_t>(base + A_OFF);
        const auto bp = local_as<int8_t>(base + B_OFF);
        int8_t* al = ap.get();
        uint32_t* bw = local_as<uint32_t>(base + B_OFF).get();
        float* xsl = local_as<float>(base + XS_OFF).get();
        [[maybe_unused]] float* xul = local_as<float>(base + XU_OFF).get();
        [[maybe_unused]] float* wml = local_as<float>(base + WM_OFF).get();
        float* wsl = local_as<float>(base + WS_OFF).get();

        float acc[TPS][IM];
        for (int t = 0; t < TPS; ++t)
            for (int i = 0; i < IM; ++i) acc[t][i] = 0.0f;
        mx::joint_matrix<sycl::sub_group, int32_t, mx::use::accumulator, IM, IN> ci;

        const int64_t kb = K / 32;   // activation scales a row
        for (int64_t k0 = 0; k0 < K; k0 += KC) {
            sycl::group_barrier(it.get_group());   // the last step's tiles are read
            const bool last = TAIL && k0 + KC > K;   // TAIL: a last step past K, zeros past it
            for (int v = lid; v < WM * (KC / 16); v += lanes) {
                const int rr = v / (KC / 16), c = v % (KC / 16) * 16;
                const int64_t row = m0 + rr;
                u32x4 val = {0u, 0u, 0u, 0u};
                if (row < rows && (!last || k0 + c < K)) val = *reinterpret_cast<const u32x4*>(xq + (row0 + row) * K + k0 + c);
                if constexpr (HALF) {   // copy h keeps the half h of each block, zeros in the other
                    const u32x4 z = {0u, 0u, 0u, 0u};
                    const bool hi = c % 32 >= 16;
                    *reinterpret_cast<u32x4*>(al + static_cast<ptrdiff_t>(rr) * LDA + c) = hi ? z : val;
                    *reinterpret_cast<u32x4*>(al + static_cast<ptrdiff_t>(WM + rr) * LDA + c) = hi ? val : z;
                } else {
                    *reinterpret_cast<u32x4*>(al + static_cast<ptrdiff_t>(rr) * LDA + c) = val;
                }
            }
            for (int v = lid; v < WM * NB; v += lanes) {
                const int b = v / WM, rr = v % WM;
                const int64_t row = m0 + rr;
                xsl[v] = row < rows && (!last || k0 / 32 + b < kb) ? xs[(row0 + row) * kb + k0 / 32 + b] : 0.0f;
                if constexpr (MIN) xul[v] = row < rows && (!last || k0 / 32 + b < kb) ? xu[(row0 + row) * kb + k0 / 32 + b] : 0.0f;
            }
            for (int v = lid; v < WN * NB; v += lanes) {
                const int nn = v % WN, b = v / WN;   // neighbouring work-items: neighbouring outputs of a VNNI row
                const Sub d = !last || k0 / 32 + b < kb ? Dec<TY>::get(wx + (size_t) (n0 + nn) * row_bytes, (int) (k0 / 32) + b, row_bytes)
                                                : Sub{{}, 0.0f, 0.0f, 0.0f};   // past K: zeros
                for (int j = 0; j < 8; ++j) bw[(b * 8 + j) * WN + nn] = d.v[j];
                wsl[(2 * b) * WN + nn] = d.s0;
                wsl[(2 * b + 1) * WN + nn] = d.s1;
                if constexpr (MIN) wml[b * WN + nn] = d.m;
            }
            sycl::group_barrier(it.get_group());
            for (int b = 0; b < NB; ++b) {
                mx::joint_matrix<sycl::sub_group, int8_t, mx::use::b, IK, IN, mx::layout::ext_intel_packed> bm;
                mx::joint_matrix_load(sg, bm, bp + static_cast<int>(b * 8 * LDBV + cs * 4), LDBV);
                for (int h = 0; h < (HALF ? 2 : 1); ++h) {
                    const float ws = wsl[(2 * b + h) * WN + cs + lane];   // this lane's column's
                    for (int t = 0; t < TPS; ++t) {
                        mx::joint_matrix<sycl::sub_group, int8_t, mx::use::a, IM, IK, mx::layout::row_major> a;
                        mx::joint_matrix_load(sg, a, ap + static_cast<int>((h * WM + rs + t * IM) * LDA + b * 32), LDA);
                        mx::joint_matrix_fill(sg, ci, 0);
                        mx::joint_matrix_mad(sg, ci, a, bm, ci);
                        const float* xr = xsl + static_cast<ptrdiff_t>(b * WM + rs + t * IM);
                        int i = 0;
                        mx::joint_matrix_apply(sg, ci, [&](int32_t& c) {
                            acc[t][i] += ws * xr[i] * exact_float(c);
                            ++i;
                        });
                    }
                }
                if constexpr (MIN) {   // less the weights' min of the lane's column times each row's sum
                    const float wmn = wml[b * WN + cs + lane];
                    for (int t = 0; t < TPS; ++t)
                        for (int i = 0; i < IM; ++i) acc[t][i] -= wmn * xul[b * WM + rs + t * IM + i];
                }
            }
        }
        for (int t = 0; t < TPS; ++t)
            for (int i = 0; i < IM; ++i) {
                const int64_t row = m0 + rs + (int64_t) t * IM + i;
                if (row < rows) dst[(int64_t) ids[row0 + row] * ld_dst + n0 + cs + lane] = acc[t][i];
            }
#else
        (void) it;
#endif
    }

    STRATA_AMD_REQD_WG(NSG * SG)
    void operator()(sycl::nd_item<1> it) const {
        if constexpr (DP4A) {
            dp4a_body(it);
        } else if constexpr (IS != 0) {
            intel_body(it);
        } else if constexpr (!built<TY>()) {
            (void) it;
        } else {
#if STRATA_IQ_MMQ_MATRIX
            const int64_t g = (int64_t) it.get_group(0), per = tiles_m * tiles_n, e = g / per, r = g % per;
            const int64_t m0 = r % tiles_m * WM, n0 = r / tiles_m * WN;
            const int64_t row0 = bounds[e], rows = bounds[e + 1] - row0;
            if (m0 >= rows) return;   // the whole work-group: this expert has fewer rows than the launch
            const auto sg = it.get_sub_group();
            const int sgid = (int) sg.get_group_linear_id(), lid = (int) it.get_local_id(0);
            const int lane = (int) sg.get_local_linear_id();
            constexpr int lanes = NSG * SG;
            const int ms = sgid / NSGN * SGM * TM, ns = sgid % NSGN * SGN * TN;
            const uint8_t* wx = wtab != nullptr ? wtab[e] + woff : w + (size_t) e * expert_bytes;
            const auto base = lm.template get_multi_ptr<sycl::access::decorated::no>();
            const auto ap = local_as<int8_t>(base + A_OFF);
            const auto bp = local_as<int8_t>(base + B_OFF);
            const auto xsp = local_as<float>(base + XS_OFF);
            const auto wsp = local_as<float>(base + WS_OFF);
            const auto stp = local_as<float>(base + ST_OFF) + static_cast<int>(sgid * TM * TN);
            int8_t* al = ap.get();
            int8_t* bl = bp.get();
            float* xsl = xsp.get();
            const auto xup = local_as<float>(base + XU_OFF);
            const auto wmp = local_as<float>(base + WM_OFF);
            [[maybe_unused]] float* xul = xup.get();
            [[maybe_unused]] float* wml = wmp.get();
            float* wsl = wsp.get();

            mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, TM, TN> acc[SGM][SGN];
            mx::joint_matrix<sycl::sub_group, int32_t, mx::use::accumulator, TM, TN> ci[SGM][SGN];
            for (int i = 0; i < SGM; ++i)
                for (int j = 0; j < SGN; ++j) mx::joint_matrix_fill(sg, acc[i][j], 0.0f);

            const int64_t kb = K / 32;   // activation scales a row
            for (int64_t k0 = 0; k0 < K; k0 += KC) {
                sycl::group_barrier(it.get_group());   // the last step's tiles are read
                const bool last = TAIL && k0 + KC > K;   // TAIL: a last step past K, zeros past it
                for (int v = lid; v < WM * (KC / 16); v += lanes) {
                    const int rr = v / (KC / 16), c = v % (KC / 16) * 16;
                    const int64_t row = m0 + rr;
                    u32x4 val = {0u, 0u, 0u, 0u};
                    if (row < rows && (!last || k0 + c < K)) val = *reinterpret_cast<const u32x4*>(xq + (row0 + row) * K + k0 + c);
                    *reinterpret_cast<u32x4*>(al + static_cast<int>(rr * LDA + c)) = val;
                }
                for (int v = lid; v < WM * NB; v += lanes) {
                    const int b = v / WM, rr = v % WM;
                    const int64_t row = m0 + rr;
                    xsl[v] = row < rows && (!last || k0 / 32 + b < kb) ? xs[(row0 + row) * kb + k0 / 32 + b] : 0.0f;
                    if constexpr (MIN) xul[v] = row < rows && (!last || k0 / 32 + b < kb) ? xu[(row0 + row) * kb + k0 / 32 + b] : 0.0f;
                }
                for (int v = lid; v < WN * NB; v += lanes) {
                    const int nn = v / NB, b = v % NB;
                    const Sub d = !last || k0 / 32 + b < kb ? Dec<TY>::get(wx + (size_t) (n0 + nn) * row_bytes, (int) (k0 / 32) + b, row_bytes)
                                                : Sub{{}, 0.0f, 0.0f, 0.0f};   // past K: zeros
                    u32x4* o = reinterpret_cast<u32x4*>(bl + static_cast<int>(nn * LDB + b * 32));
                    o[0] = u32x4{d.v[0], d.v[1], d.v[2], d.v[3]};
                    o[1] = u32x4{d.v[4], d.v[5], d.v[6], d.v[7]};
                    wsl[(2 * b) * WN + nn] = d.s0;
                    wsl[(2 * b + 1) * WN + nn] = d.s1;
                    if constexpr (MIN) wml[b * WN + nn] = d.m;
                }
                sycl::group_barrier(it.get_group());
                for (int b = 0; b < NB; ++b) {
                    mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, TM, TN> xsf[SGM];
                    for (int i = 0; i < SGM; ++i)   // element (r, c): the scale of row r
                        mx::joint_matrix_load(sg, xsf[i], xsp + static_cast<int>(b * WM + ms + i * TM), 0, mx::layout::col_major);
                    for (int h = 0; h < (HALF ? 2 : 1); ++h) {
                        for (int i = 0; i < SGM; ++i)
                            for (int j = 0; j < SGN; ++j) mx::joint_matrix_fill(sg, ci[i][j], 0);
                        for (int kk = HALF ? h * TK : 0; kk < (HALF ? h * TK + TK : 32); kk += TK) {
                            mx::joint_matrix<sycl::sub_group, int8_t, mx::use::a, TM, TK, mx::layout::row_major> a[SGM];
                            mx::joint_matrix<sycl::sub_group, int8_t, mx::use::b, TK, TN, mx::layout::col_major> bm[SGN];
                            for (int i = 0; i < SGM; ++i)
                                mx::joint_matrix_load(sg, a[i], ap + static_cast<int>((ms + i * TM) * LDA + b * 32 + kk), LDA);
                            for (int j = 0; j < SGN; ++j)
                                mx::joint_matrix_load(sg, bm[j], bp + static_cast<int>((ns + j * TN) * LDB + b * 32 + kk), LDB);
                            for (int i = 0; i < SGM; ++i)
                                for (int j = 0; j < SGN; ++j) mx::joint_matrix_mad(sg, ci[i][j], a[i], bm[j], ci[i][j]);
                        }
                        for (int j = 0; j < SGN; ++j)
                            for (int i = 0; i < SGM; ++i) {
                                // the pair of scales (r, c): the weights' of column c along each row, times the row's
                                mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, TM, TN> sc;
                                mx::joint_matrix_load(sg, sc, wsp + static_cast<int>((2 * b + h) * WN + ns + j * TN), 0,
                                                      mx::layout::row_major);
                                mx::joint_matrix_apply(sg, sc, xsf[i], [](float& s, float& x) { s *= x; });
                                mx::joint_matrix_apply(sg, sc, ci[i][j], [](float& s, int32_t& c) { s *= exact_float(c); });
                                mx::joint_matrix_apply(sg, acc[i][j], sc, [](float& a, float& s) { a += s; });
                            }
                    }
                    if constexpr (MIN)   // less the weights' min of column c times row r's sum
                        for (int i = 0; i < SGM; ++i) {
                            mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, TM, TN> xuf;
                            mx::joint_matrix_load(sg, xuf, xup + static_cast<int>(b * WM + ms + i * TM), 0,
                                                  mx::layout::col_major);
                            for (int j = 0; j < SGN; ++j) {
                                mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, TM, TN> mn;
                                mx::joint_matrix_load(sg, mn, wmp + static_cast<int>(b * WN + ns + j * TN), 0,
                                                      mx::layout::row_major);
                                mx::joint_matrix_apply(sg, mn, xuf, [](float& m, float& x) { m *= x; });
                                mx::joint_matrix_apply(sg, acc[i][j], mn, [](float& a, float& m) { a -= m; });
                            }
                        }
                }
            }
            // out through local memory, the rows past the expert's last not written
            for (int i = 0; i < SGM; ++i)
                for (int j = 0; j < SGN; ++j) {
                    mx::joint_matrix_store(sg, acc[i][j], stp, TN, mx::layout::row_major);
                    sycl::group_barrier(sg);
                    const int64_t rb = m0 + ms + (int64_t) i * TM, cb = n0 + ns + (int64_t) j * TN;
                    for (int q = lane; q < TM * TN; q += SG) {
                        const int64_t row = rb + q / TN;
                        if (row < rows) dst[(int64_t) ids[row0 + row] * ld_dst + cb + q % TN] = stp[q];
                    }
                    sycl::group_barrier(sg);
                }
#elif STRATA_IQ_MMQ_WMMA
            // The int8 WMMA of RDNA3 (GFX11) and RDNA4 (GFX12), 16 x 16 x 16 on 32 lanes; lane l holds column l % 16 of
            // B and of the accumulator, row l % 16 of A.  GFX12: each lane 8 of the 16 K values (l / 16 picks which),
            // the accumulator's rows 8 (l / 16) + e.  GFX11: each lane all 16 K values (lanes 16-31 repeat 0-15), the
            // accumulator's rows 2 e + l / 16.  The same products and scales as the joint_matrix body above, in the
            // positions those layouts give.
            const int64_t g = (int64_t) it.get_group(0), per = tiles_m * tiles_n, e = g / per, r = g % per;
            const int64_t m0 = r % tiles_m * WM, n0 = r / tiles_m * WN;
            const int64_t row0 = bounds[e], rows = bounds[e + 1] - row0;
            if (m0 >= rows) return;   // the whole work-group: this expert has fewer rows than the launch
            const auto sg = it.get_sub_group();
            const int sgid = (int) sg.get_group_linear_id(), lid = (int) it.get_local_id(0);
            const int lane = (int) sg.get_local_linear_id(), lo = lane & 15, hi = lane >> 4;
            constexpr int lanes = NSG * SG;
            const int ms = sgid / NSGN * SGM * TM, ns = sgid % NSGN * SGN * TN;
            // The weights' and activations' pointers cast to the global address space (XeStrata-890M): the kernel's
            // arguments are generic pointers, which AMD's compiler read with flat loads; global loads instead (no
            // measured change on the 890M: each decode still waits on its own loads, the grid lookups behind the
            // block's)
            const auto global_of = [](auto p) {
                return sycl::address_space_cast<sycl::access::address_space::global_space,
                                                sycl::access::decorated::no>(p).get();
            };
            const uint8_t* wx = global_of(wtab != nullptr ? wtab[e] + woff : w + (size_t) e * expert_bytes);
            const int8_t* xqg = global_of(xq);
            const float* xsg = global_of(xs);
            [[maybe_unused]] const float* xug = MIN ? global_of(xu) : nullptr;
            const auto base = lm.template get_multi_ptr<sycl::access::decorated::no>();
            int8_t* al = local_as<int8_t>(base + A_OFF).get();
            int8_t* bl = local_as<int8_t>(base + B_OFF).get();
            float* xsl = local_as<float>(base + XS_OFF).get();
            [[maybe_unused]] float* xul = local_as<float>(base + XU_OFF).get();
            [[maybe_unused]] float* wml = local_as<float>(base + WM_OFF).get();
            float* wsl = local_as<float>(base + WS_OFF).get();
            using i8v = int __attribute__((ext_vector_type(8)));
#if defined(__GFX12__)
            using frag = int __attribute__((ext_vector_type(2)));
            constexpr int KOFF = 8;   // bytes of K a lane holds past its half's start
            auto row_of = [&](int el) { return hi * 8 + el; };
#else
            using frag = int __attribute__((ext_vector_type(4)));
            constexpr int KOFF = 0;
            auto row_of = [&](int el) { return el * 2 + hi; };
#endif
            float acc[SGM][SGN][8];
            for (int i = 0; i < SGM; ++i)
                for (int j = 0; j < SGN; ++j)
                    for (int el = 0; el < 8; ++el) acc[i][j][el] = 0.0f;

            [[maybe_unused]] const typename GridOf<TY>::type* gl = nullptr;
            if constexpr (GRID_BYTES > 0) {
                u32x4* g = reinterpret_cast<u32x4*>(local_as<uint8_t>(base + GRID_OFF).get());
                const u32x4* t = reinterpret_cast<const u32x4*>(Dec<TY>::table());
                for (int v = lid; v < GRID_BYTES / 16; v += lanes) g[v] = t[v];
                gl = reinterpret_cast<const typename GridOf<TY>::type*>(g);
            }
            const int64_t kb = K / 32;   // activation scales a row
            for (int64_t k0 = 0; k0 < K; k0 += KC) {
                sycl::group_barrier(it.get_group());   // the last step's tiles are read
                const bool last = TAIL && k0 + KC > K;   // TAIL: a last step past K, zeros past it
                // the step's loads (XeStrata-890M): the counts a lane are compile-time, the loops unrolled, and the
                // weights' bytes are all loaded before any block is decoded (Dec::load, Dec::decode)
                constexpr int AV = WM * (KC / 16), SV = WM * NB, DV = WN * NB;
                constexpr int APL = (AV + lanes - 1) / lanes, SPL = (SV + lanes - 1) / lanes, DPL = (DV + lanes - 1) / lanes;
#pragma unroll
                for (int u = 0; u < APL; ++u) {
                    const int v = lid + u * lanes;
                    if (AV % lanes != 0 && v >= AV) break;
                    const int rr = v / (KC / 16), c = v % (KC / 16) * 16;
                    const int64_t row = m0 + rr;
                    u32x4 val = {0u, 0u, 0u, 0u};
                    if (row < rows && (!last || k0 + c < K)) val = *reinterpret_cast<const u32x4*>(xqg + (row0 + row) * K + k0 + c);
                    *reinterpret_cast<u32x4*>(al + rr * LDA + c) = val;
                }
#pragma unroll
                for (int u = 0; u < SPL; ++u) {
                    const int v = lid + u * lanes;
                    if (SV % lanes != 0 && v >= SV) break;
                    const int b = v / WM, rr = v % WM;
                    const int64_t row = m0 + rr;
                    xsl[v] = row < rows && (!last || k0 / 32 + b < kb) ? xsg[(row0 + row) * kb + k0 / 32 + b] : 0.0f;
                    if constexpr (MIN) xul[v] = row < rows && (!last || k0 / 32 + b < kb) ? xug[(row0 + row) * kb + k0 / 32 + b] : 0.0f;
                }
                if constexpr (split_dec<Dec<TY>>::value) {
                    typename Dec<TY>::Raw raw[DPL];
#pragma unroll
                    for (int u = 0; u < DPL; ++u) {
                        const int v = lid + u * lanes;
                        if (DV % lanes != 0 && v >= DV) break;
                        const int nn = v / NB, b = v % NB;
                        if (!last || k0 / 32 + b < kb)
                            raw[u] = Dec<TY>::load(wx + (size_t) (n0 + nn) * row_bytes, (int) (k0 / 32) + b, row_bytes);
                    }
#pragma unroll
                    for (int u = 0; u < DPL; ++u) {
                        const int v = lid + u * lanes;
                        if (DV % lanes != 0 && v >= DV) break;
                        const int nn = v / NB, b = v % NB;
                        const Sub d = !last || k0 / 32 + b < kb
                                          ? Dec<TY>::decode(raw[u], wx + (size_t) (n0 + nn) * row_bytes, (int) (k0 / 32) + b, row_bytes, gl)
                                          : Sub{{}, 0.0f, 0.0f, 0.0f};   // past K: zeros
                        u32x4* o = reinterpret_cast<u32x4*>(bl + nn * LDB + b * 32);
                        o[0] = u32x4{d.v[0], d.v[1], d.v[2], d.v[3]};
                        o[1] = u32x4{d.v[4], d.v[5], d.v[6], d.v[7]};
                        wsl[(2 * b) * WN + nn] = d.s0;
                        wsl[(2 * b + 1) * WN + nn] = d.s1;
                        if constexpr (MIN) wml[b * WN + nn] = d.m;
                    }
                } else {
#pragma unroll
                    for (int u = 0; u < DPL; ++u) {
                        const int v = lid + u * lanes;
                        if (DV % lanes != 0 && v >= DV) break;
                        const int nn = v / NB, b = v % NB;
                        const Sub d = !last || k0 / 32 + b < kb ? Dec<TY>::get(wx + (size_t) (n0 + nn) * row_bytes, (int) (k0 / 32) + b, row_bytes)
                                                    : Sub{{}, 0.0f, 0.0f, 0.0f};   // past K: zeros
                        u32x4* o = reinterpret_cast<u32x4*>(bl + nn * LDB + b * 32);
                        o[0] = u32x4{d.v[0], d.v[1], d.v[2], d.v[3]};
                        o[1] = u32x4{d.v[4], d.v[5], d.v[6], d.v[7]};
                        wsl[(2 * b) * WN + nn] = d.s0;
                        wsl[(2 * b + 1) * WN + nn] = d.s1;
                        if constexpr (MIN) wml[b * WN + nn] = d.m;
                    }
                }
                sycl::group_barrier(it.get_group());
                for (int b = 0; b < NB; ++b) {
                    // The sub-group's rows one tile at a time (XeStrata-890M): its int32 tiles, the rows' scales
                    // and the A fragment of one row tile at a time, B's fragments loaded again for each (the 64 x 128
                    // tile's kernel spilled 17-26 registers at the backend's 128 with the two row tiles' int32 tiles
                    // and scales live together), and the scales applied with one fma (ws xsr) c + acc: three
                    // instructions an element instead of four
                    for (int h = 0; h < (HALF ? 2 : 1); ++h)
                        for (int i = 0; i < SGM; ++i) {
                            float xsr[8];   // the rows' scales of the block, in the accumulator's row order
                            for (int el = 0; el < 8; ++el) xsr[el] = xsl[b * WM + ms + i * TM + row_of(el)];
                            i8v ci[SGN];
                            for (int j = 0; j < SGN; ++j) ci[j] = i8v{0, 0, 0, 0, 0, 0, 0, 0};
                            for (int kk = HALF ? h * TK : 0; kk < (HALF ? h * TK + TK : 32); kk += TK) {
                                const frag a = *reinterpret_cast<const frag*>(al + (ms + i * TM + lo) * LDA + b * 32 + kk + hi * KOFF);
                                for (int j = 0; j < SGN; ++j) {
                                    const frag bm = *reinterpret_cast<const frag*>(bl + (ns + j * TN + lo) * LDB + b * 32 + kk + hi * KOFF);
#if defined(__GFX12__)
                                    ci[j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a, true, bm, ci[j], false);
#else
                                    ci[j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, bm, ci[j], false);
#endif
                                }
                            }
                            for (int j = 0; j < SGN; ++j) {
                                const float ws = wsl[(2 * b + h) * WN + ns + j * TN + lo];   // this lane's column's
                                for (int el = 0; el < 8; ++el)
                                    // (float) c: one full-rate conversion on RDNA (exact below 2^24, as exact_float)
                                    acc[i][j][el] = sycl::fma(ws * xsr[el], (float) ci[j][el], acc[i][j][el]);
                            }
                        }
                    if constexpr (MIN)
                        for (int j = 0; j < SGN; ++j) {
                            const float wm = wml[b * WN + ns + j * TN + lo];
                            for (int i = 0; i < SGM; ++i)
                                for (int el = 0; el < 8; ++el)
                                    acc[i][j][el] -= wm * xul[b * WM + ms + i * TM + row_of(el)];
                        }
                }
            }
            // straight out, the rows past the expert's last not written
            for (int i = 0; i < SGM; ++i)
                for (int j = 0; j < SGN; ++j) {
                    const int64_t cb = n0 + ns + (int64_t) j * TN + lo;
                    for (int el = 0; el < 8; ++el) {
                        const int64_t row = m0 + ms + (int64_t) i * TM + row_of(el);
                        if (row < rows) dst[(int64_t) ids[row0 + row] * ld_dst + cb] = acc[i][j][el];
                    }
                }
#endif
        }
    }
    auto get(syclex::properties_tag) const {
        // Intel's XMX bodies in the large register file: in the default one their sums and tiles spilled, 3.1x slower
        // (B70, llm-jp-4.1's experts, 256 rows an expert: 6,464 against 2,070 us)
        if constexpr (DP4A) return syclex::properties{};   // no sub-group operations: the device's own size
        else if constexpr (IS != 0)
            return syclex::properties{syclex::sub_group_size<STRATA_SUB_GROUP(ISG)>,
                                      sycl::ext::intel::experimental::grf_size<256>};
        else return syclex::properties{syclex::sub_group_size<STRATA_SUB_GROUP(ISG)>};
    }
};

template<int TY, int SGM, int NSGM, bool DP4A, bool TAIL, int IS>
sycl::event launch(sycl::queue& q, const void* w, const uint8_t* const* wtab, size_t woff, size_t expert_bytes,
                   int64_t w_rows, int64_t w_cols, int n, const void* xq, int64_t total_rows, const int32_t* bounds,
                   int64_t max_rows, const int32_t* ids, float* dst, int64_t ld_dst) {
    using KT = Kernel<TY, SGM, NSGM, DP4A, TAIL, IS>;
    const int64_t tiles_m = (max_rows + KT::WM - 1) / KT::WM, tiles_n = w_rows / KT::WN;
    const size_t row_bytes = (size_t) (w_cols / Dec<TY>::QK) * (size_t) Dec<TY>::BSZ;
    const auto* x8 = static_cast<const int8_t*>(xq);
    const auto* xs = reinterpret_cast<const float*>(x8 + total_rows * w_cols);
    const auto* xu = xs + total_rows * (w_cols / 32);
    // RDNA's WMMA body does without the output staging tile (LOCAL_DIRECT)
    const size_t local = !DP4A && xe::amd_wmma(q.get_device()) ? (size_t) KT::LOCAL_DIRECT_GRID : (size_t) KT::LOCAL;
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<uint8_t, 1> lm(sycl::range<1>(local), h);
        const size_t wg = (size_t) KT::NSG * SG;
        h.parallel_for(sycl::nd_range<1>((size_t) n * (size_t) (tiles_m * tiles_n) * wg, wg),
                       KT{static_cast<const uint8_t*>(w), wtab, woff, expert_bytes, row_bytes, x8, xs, xu, w_cols, bounds, ids, dst,
                          ld_dst, tiles_m, tiles_n, lm});
    });
}

// Whether an int32 accumulator of Intel's 8 x N shape holds, in lane l, column l with element i in row i (the order
// intel_body applies the scales in): 1 when it does.
template<int IS>
struct IntelLayoutCheck {
    static constexpr int N = IS == 1 ? 16 : 8;
    sycl::local_accessor<int32_t, 1> pi;
    int* out;
    void operator()(sycl::nd_item<1> it) const {
#if STRATA_IQ_MMQ_INTEL
        const auto sg = it.get_sub_group();
        const int lid = (int) it.get_local_id(0);
        for (int i = lid; i < 8 * N; i += N) pi[i] = i;
        sycl::group_barrier(it.get_group());
        mx::joint_matrix<sycl::sub_group, int32_t, mx::use::accumulator, 8, N> a;
        mx::joint_matrix_load(sg, a, pi.template get_multi_ptr<sycl::access::decorated::no>(), N, mx::layout::row_major);
        int bad = 0, i = 0;
        mx::joint_matrix_apply(sg, a, [&](int32_t& v) { bad |= v != i * N + lid; ++i; });
        bad |= i != 8;
        bad = sycl::reduce_over_group(sg, bad, sycl::bit_or<int>());
        if (lid == 0) *out = bad ? 0 : 1;
#else
        (void) it;
#endif
    }
    auto get(syclex::properties_tag) const { return syclex::properties{syclex::sub_group_size<STRATA_SUB_GROUP(N)>}; }
};

// Intel's int8 XMX shape the device reports at its sub-group size, its local memory and work-group size permitting
// intel_body, and its accumulators laid out as intel_body reads them: 1 Xe2's (8 x 16 x 32 on 16 lanes), 2 Xe-HPG's (8
// x 8 x 32 on 8), else 0.  STRATA_NO_XMX=1: 0.
template<int IS>
bool intel_fits(sycl::queue& q) {
    const sycl::device d = q.get_device();
    using Big = Kernel<17, 2, 2, false, false, IS>;
    const size_t local = std::max<size_t>(Big::LOCAL, Kernel<12, 2, 2, false, false, IS>::LOCAL);
    if (d.get_info<sycl::info::device::local_mem_size>() < local ||
        d.get_info<sycl::info::device::max_work_group_size>() < (size_t) Big::NSG * SG)
        return false;
    try {   // the large register file (intel_body's properties), as xmx_gemm.cpp's large_grf_ok checks it
        (void) sycl::get_kernel_bundle<sycl::bundle_state::executable>(q.get_context(), {d},
                                                                       {sycl::get_kernel_id<Big>()});
        return true;
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "strata: the int8 XMX products do not build for this GPU (%s)\n", e.what());
        return false;
    }
}
template<int IS>
bool intel_layout_ok(sycl::queue& q) {
    int* out = sycl::malloc_device<int>(1, q);
    if (out == nullptr) return false;
    int res = 0;
    try {
        q.submit([&](sycl::handler& h) {
            sycl::local_accessor<int32_t, 1> pi(sycl::range<1>((size_t) 8 * IntelLayoutCheck<IS>::N), h);
            h.parallel_for(sycl::nd_range<1>(IntelLayoutCheck<IS>::N, IntelLayoutCheck<IS>::N), IntelLayoutCheck<IS>{pi, out});
        });
        q.memcpy(&res, out, sizeof(int)).wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "strata: the int8 XMX layout check did not run (%s)\n", e.what());
        res = 0;
    }
    sycl::free(out, q);
    return res == 1;
}
int intel_shape_of(sycl::queue& q) {
    static core::PerDevice<int> per_device;
    return per_device.get(q.get_device(), [&q] {
        const char* no = std::getenv("STRATA_NO_XMX");
        if (no != nullptr && std::strtol(no, nullptr, 10) != 0) return 0;
        const sycl::device d = q.get_device();
        const auto sgs = d.get_info<sycl::info::device::sub_group_sizes>();
        const auto combos = matrix_combinations(d);
        for (int is = 1; is <= 2; ++is) {
            const size_t n = is == 1 ? 16 : 8;
            if (std::find(sgs.begin(), sgs.end(), n) == sgs.end()) continue;
            bool rep = false;
            for (const auto& c : combos) {
                const bool m = c.msize == 8 || (c.msize == 0 && c.max_msize >= 8);
                const bool nn = c.nsize == n || (c.nsize == 0 && c.max_nsize >= n);
                const bool k = c.ksize == 32 || (c.ksize == 0 && c.max_ksize >= 32);
                rep = rep || (c.atype == mx::matrix_type::sint8 && c.btype == mx::matrix_type::sint8 &&
                              c.ctype == mx::matrix_type::sint32 && c.dtype == mx::matrix_type::sint32 && m && nn && k);
            }
            if (!rep || !(is == 1 ? intel_fits<1>(q) : intel_fits<2>(q))) continue;
            if (is == 1 ? intel_layout_ok<1>(q) : intel_layout_ok<2>(q)) return is;
            std::fprintf(stderr, "strata: the GPU's int8 XMX accumulators are laid out otherwise: the experts' prompt "
                                 "products do not take them\n");
        }
        return 0;
    });
}

template<int TY>
sycl::event by_rows(sycl::queue& q, const void* w, const uint8_t* const* wtab, size_t woff, size_t expert_bytes,
                    int64_t w_rows, int64_t w_cols, int n, const void* xq, int64_t total_rows, const int32_t* bounds,
                    int64_t max_rows, const int32_t* ids, float* dst, int64_t ld_dst, int64_t rows) {
#define STRATA_MMQ_LAUNCH(SGM, NSGM, DP4A, TAIL, IS)                                                                 \
    launch<TY, SGM, NSGM, DP4A, TAIL, IS>(q, w, wtab, woff, expert_bytes, w_rows, w_cols, n, xq, total_rows, bounds, \
                                          max_rows, ids, dst, ld_dst)
#define STRATA_MMQ_ROWS(DP4A, TAIL, IS)                                                        \
    do {                                                                                        \
        if (max_rows <= 16) return STRATA_MMQ_LAUNCH(1, 1, DP4A, TAIL, IS);                     \
        if (max_rows <= 32 || mean_rows <= 32) return STRATA_MMQ_LAUNCH(2, 1, DP4A, TAIL, IS);  \
        return STRATA_MMQ_LAUNCH(2, 2, DP4A, TAIL, IS);                                         \
    } while (0)
    // the tile's rows by the group's largest, but 32 where the experts average no more: a 64-row tile's work-groups
    // then multiply mostly empty rows, and the extra tiles of the few larger experts cost less (Ornith GSQ-RCO 3bit's
    // 515-token prompt, 16 rows an expert: RX 9060 XT +8%; its 4K prompt, 111 rows, keeps the 64-row tile)
    const int64_t mean_rows = rows / std::max(n, 1);
    // dp4a where the matrix form does not run (iq_mmq_grouped is called where one of the two does); the K-tail form
    // only for a K that is no multiple of the step, so that the others keep their loads unguarded
    const bool tail = w_cols % KC != 0;
    if (!iq_mmq_usable(q)) {
        if (tail) STRATA_MMQ_ROWS(true, true, 0);
        STRATA_MMQ_ROWS(true, false, 0);
    }
    // Intel's XMX: the 32-row tile's work-group (128 lanes) decodes the step's weights at half the 64-row one's rate,
    // and took longer for 32 rows than the 64-row tile for 64 on the B70; so 16 rows, else 64
#define STRATA_MMQ_ROWS_INTEL(TAIL, IS)                                                     \
    do {                                                                                    \
        if (max_rows <= 16) return STRATA_MMQ_LAUNCH(1, 1, false, TAIL, IS);                \
        return STRATA_MMQ_LAUNCH(2, 2, false, TAIL, IS);                                    \
    } while (0)
    if (const int is = intel_shape_of(q); is == 1) {
        if (tail) STRATA_MMQ_ROWS_INTEL(true, 1);
        STRATA_MMQ_ROWS_INTEL(false, 1);
    } else if (is == 2) {
        if (tail) STRATA_MMQ_ROWS_INTEL(true, 2);
        STRATA_MMQ_ROWS_INTEL(false, 2);
    }
#undef STRATA_MMQ_ROWS_INTEL
    // XeStrata-890M: a dense projection (one expert of many rows: prefill.cpp's dense_proj) takes a 128-row tile of
    // 8 sub-groups (4 row tiles by 2 column tiles each): a step's decoded weights serve twice the rows.  iq_mmq_parity
    // --dense, IQ4_XS, 2048 rows: 10240 x 2560 17.5 -> 14.0 ms (the FP16 WMMA 19.1), 6144 x 2560 10.9 -> 8.4
    // (11.7), 2560 x 6144 13.8 -> 9.3 (10.6); 512 rows of 10240 x 2560 3.7 -> 2.9 (5.4).  256-row tiles and 512-lane
    // work-groups lost (2 x 4: 16.9 ms, 4 x 4: 18.8, 8 x 1: 33.6).  STRATA_MMQ_BIG=0 keeps the 64-row tile, =1 gives
    // the 128-row one to the experts' groups too.
    static const int big = [] { const char* v = std::getenv("STRATA_MMQ_BIG"); return v ? (int) std::strtol(v, nullptr, 10) : -1; }();
    if ((big == 1 || (big < 0 && n == 1)) && max_rows > 64 && !tail) return STRATA_MMQ_LAUNCH(4, 2, false, false, 0);
    if (tail) STRATA_MMQ_ROWS(false, true, 0);
    STRATA_MMQ_ROWS(false, false, 0);
#undef STRATA_MMQ_ROWS
#undef STRATA_MMQ_LAUNCH
}

bool env_on(const char* name) {
    const char* v = std::getenv(name);
    return v != nullptr && std::strtol(v, nullptr, 10) != 0;
}

bool shape_reported(const sycl::device& d) {
    const auto sgs = d.get_info<sycl::info::device::sub_group_sizes>();
    if (std::find(sgs.begin(), sgs.end(), (size_t) SG) == sgs.end()) return false;
    for (const auto& c : matrix_combinations(d)) {
        const bool m = c.msize == (size_t) TM || (c.msize == 0 && c.max_msize >= (size_t) TM);
        const bool n = c.nsize == (size_t) TN || (c.nsize == 0 && c.max_nsize >= (size_t) TN);
        const bool k = c.ksize == (size_t) TK || (c.ksize == 0 && c.max_ksize >= (size_t) TK);
        if (c.atype == mx::matrix_type::sint8 && c.btype == mx::matrix_type::sint8 &&
            c.ctype == mx::matrix_type::sint32 && c.dtype == mx::matrix_type::sint32 && m && n && k)
            return true;
    }
    return false;
}

// The positions a lane's int32 accumulator elements take against those of its FP32 one, walked together as the
// products walk them: 1 when every pair agrees.
struct LayoutCheck {
    sycl::local_accessor<int32_t, 1> pi;
    sycl::local_accessor<float, 1> pf;
    int* out;
    void operator()(sycl::nd_item<1> it) const {
        if constexpr (!built<0>()) {
            (void) it;
        } else {
#if STRATA_IQ_MMQ_MATRIX
            const auto sg = it.get_sub_group();
            const int lid = (int) it.get_local_id(0);
            for (int i = lid; i < TM * TN; i += SG) { pi[i] = i; pf[i] = (float) i; }
            sycl::group_barrier(it.get_group());
            mx::joint_matrix<sycl::sub_group, int32_t, mx::use::accumulator, TM, TN> a;
            mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, TM, TN> b;
            mx::joint_matrix_load(sg, a, pi.template get_multi_ptr<sycl::access::decorated::no>(), TN, mx::layout::row_major);
            mx::joint_matrix_load(sg, b, pf.template get_multi_ptr<sycl::access::decorated::no>(), TN, mx::layout::row_major);
            int bad = 0, n = 0;
            mx::joint_matrix_apply(sg, b, a, [&](float& f, int32_t& v) { bad |= (int) f != v; ++n; });
            bad |= n != NE;
            bad = sycl::reduce_over_group(sg, bad, sycl::bit_or<int>());
            if (lid == 0) *out = bad ? 0 : 1;
#endif
        }
    }
    auto get(syclex::properties_tag) const { return syclex::properties{syclex::sub_group_size<STRATA_SUB_GROUP(SG)>}; }
};

}  // namespace

bool iq_mmq_usable(sycl::queue& q) {
    static core::PerDevice<bool> per_device;
    return per_device.get(q.get_device(), [&q] {
        if (env_on("STRATA_NO_XMX")) return false;
        const sycl::device d = q.get_device();
        const bool wmma = xe::amd_wmma(d);   // its int8 WMMA: the STRATA_IQ_MMQ_WMMA body
        if (!wmma && !shape_reported(d)) return intel_shape_of(q) != 0;   // Intel's XMX shapes (intel_body)
        using Big = Kernel<16, 2, 2>;
        if (d.get_info<sycl::info::device::local_mem_size>() < (size_t) Big::LOCAL ||
            d.get_info<sycl::info::device::max_work_group_size>() < (size_t) Big::NSG * SG)
            return false;
        if (wmma) return true;   // its layout is the WMMA instructions' own, written out in the kernel
        int* out = sycl::malloc_device<int>(1, q);
        if (out == nullptr) return false;
        int res = 0;
        try {
            q.submit([&](sycl::handler& h) {
                sycl::local_accessor<int32_t, 1> pi(sycl::range<1>((size_t) TM * TN), h);
                sycl::local_accessor<float, 1> pf(sycl::range<1>((size_t) TM * TN), h);
                h.parallel_for(sycl::nd_range<1>(SG, SG), LayoutCheck{pi, pf, out});
            });
            q.memcpy(&res, out, sizeof(int)).wait();
        } catch (const sycl::exception& e) {
            std::fprintf(stderr, "strata: the int8 expert products' layout check did not run (%s)\n", e.what());
            res = 0;
        }
        sycl::free(out, q);
        if (!res) std::fprintf(stderr, "strata: the GPU's int8 and FP32 accumulators differ in layout: the experts' "
                               "prompt products stay in FP16\n");
        return res == 1;
    });
}

bool iq_mmq_dp4a_usable(sycl::queue& q) {
    static core::PerDevice<bool> per_device;
    return per_device.get(q.get_device(), [&q] {
        const sycl::device d = q.get_device();
        using Big = Kernel<16, 2, 2, true>;
        return d.get_info<sycl::info::device::local_mem_size>() >= (size_t) Big::LOCAL &&
               d.get_info<sycl::info::device::max_work_group_size>() >= (size_t) Big::NSG * SG;
    });
}

const char* iq_mmq_path(sycl::queue& q) {
    if (iq_mmq_usable(q)) {
        const sycl::device d = q.get_device();
        return xe::amd_wmma(d) ? "WMMA" : shape_reported(d) ? "joint_matrix" : "XMX";
    }
    return iq_mmq_dp4a_usable(q) ? "DP4a" : nullptr;
}

bool iq_mmq_type_sums(int t) { return t == 12 || t == 13; }   // has_min

bool iq_mmq_type_ok(int t) {
    switch (t) {
        case 6: case 8: case 10: case 11: case 12: case 13: case 14: case kNativeQ6KRows: case kNativeQ8Rows:
        case 16: case 17: case 18: case 20: case 21: case 22: case 23: case 29: case 42: return true;
        default: return false;
    }
}

size_t iq_mmq_row_bytes(int t, int64_t cols) {
    switch (t) {
        case 6: return (size_t) (cols / Dec<6>::QK) * Dec<6>::BSZ;
        case 10: return (size_t) (cols / 256) * Dec<10>::BSZ;   // the row's own values (read twice as long)
        case 11: return (size_t) (cols / Dec<11>::QK) * Dec<11>::BSZ;
        case 12: return (size_t) (cols / Dec<12>::QK) * Dec<12>::BSZ;
        case 13: return (size_t) (cols / Dec<13>::QK) * Dec<13>::BSZ;
        case 8: return (size_t) (cols / Dec<8>::QK) * Dec<8>::BSZ;
        case 14: return (size_t) (cols / Dec<14>::QK) * Dec<14>::BSZ;
        case kNativeQ6KRows: return (size_t) (cols / Dec<kNativeQ6KRows>::QK) * Dec<kNativeQ6KRows>::BSZ;
        case kNativeQ8Rows: return (size_t) (cols / Dec<kNativeQ8Rows>::QK) * Dec<kNativeQ8Rows>::BSZ;
        case 16: return (size_t) (cols / Dec<16>::QK) * Dec<16>::BSZ;
        case 17: return (size_t) (cols / Dec<17>::QK) * Dec<17>::BSZ;
        case 18: return (size_t) (cols / Dec<18>::QK) * Dec<18>::BSZ;
        case 20: return (size_t) (cols / Dec<20>::QK) * Dec<20>::BSZ;
        case 21: return (size_t) (cols / Dec<21>::QK) * Dec<21>::BSZ;
        case 22: return (size_t) (cols / Dec<22>::QK) * Dec<22>::BSZ;
        case 23: return (size_t) (cols / Dec<23>::QK) * Dec<23>::BSZ;
        case 29: return (size_t) (cols / Dec<29>::QK) * Dec<29>::BSZ;
        case 42: return (size_t) (cols / Dec<42>::QK) * Dec<42>::BSZ;
        default: return 0;
    }
}

size_t iq_mmq_act_bytes(int64_t rows, int64_t cols) { return (size_t) (rows * cols + 2 * rows * (cols / 32) * 4); }

bool iq_mmq_type_dup(int t) { return t == 10; }

// float rows, or FP16 / BF16 rows as uint16_t (XeStrata-890M: the dense projections' 16-bit-only inputs)
template <class X, bool BF16 = false>
sycl::event quantize_rows(sycl::queue& q, const X* x, const int32_t* ids, void* xq, int64_t cols, int64_t ld,
                          int64_t rows, bool sums, bool dup) {
    // dup (Q2_K, Dec<10>): each block written twice, as a row of 2 cols values
    const int64_t oc = dup ? 2 * cols : cols, rep = dup ? 2 : 1;
    auto* x8 = static_cast<int8_t*>(xq);
    auto* xs = reinterpret_cast<float*>(x8 + rows * oc);
    const int64_t nb = cols / 32, onb = oc / 32;
    float* xu = xs + rows * onb;
    return q.parallel_for(sycl::range<1>((size_t) (rows * nb)), [=](sycl::id<1> i) {
        const int64_t r = (int64_t) i[0] / nb, b = (int64_t) i[0] % nb;
        const X* xr = x + (int64_t) (ids ? ids[r] : r) * ld + b * 32;
        float v[32], amax = 0.0f;
        // (16-byte vector loads here were no faster on the 890M: the kernel runs at the memory's rate)
        for (int j = 0; j < 32; ++j) {
            if constexpr (std::is_same_v<X, float>) v[j] = xr[j];
            else if constexpr (BF16) v[j] = sycl::bit_cast<float>((uint32_t) xr[j] << 16);
            else v[j] = (float) sycl::bit_cast<sycl::half>(xr[j]);
            amax = sycl::fmax(amax, sycl::fabs(v[j]));
        }
        const float d = amax / 127.0f, id = amax > 0.0f ? 1.0f / d : 0.0f;
        uint32_t p[8];
        int sum = 0;
        for (int j = 0; j < 8; ++j) {
            uint32_t u = 0;
            for (int k = 0; k < 4; ++k) {
                const int q = (int) sycl::round(v[4 * j + k] * id);
                sum += q;
                u |= (uint32_t) (uint8_t) (int8_t) q << (8 * k);
            }
            p[j] = u;
        }
        for (int64_t c = 0; c < rep; ++c) {
            const int64_t ob = b * rep + c;
            u32x4* o = reinterpret_cast<u32x4*>(x8 + r * oc + ob * 32);
            o[0] = u32x4{p[0], p[1], p[2], p[3]};
            o[1] = u32x4{p[4], p[5], p[6], p[7]};
            xs[r * onb + ob] = d;
            if (sums) xu[r * onb + ob] = d * (float) sum;
        }
    });
}

sycl::event iq_mmq_quantize(sycl::queue& q, const float* x, const int32_t* ids, void* xq, int64_t cols, int64_t ld,
                            int64_t rows, bool sums, bool dup) {
    return quantize_rows<float>(q, x, ids, xq, cols, ld, rows, sums, dup);
}
sycl::event iq_mmq_quantize_f16(sycl::queue& q, const uint16_t* x, void* xq, int64_t cols, int64_t ld, int64_t rows,
                                bool sums, bool dup) {
    return quantize_rows<uint16_t>(q, x, nullptr, xq, cols, ld, rows, sums, dup);
}
sycl::event iq_mmq_quantize_bf16(sycl::queue& q, const uint16_t* x, void* xq, int64_t cols, int64_t ld, int64_t rows,
                                 bool sums, bool dup) {
    return quantize_rows<uint16_t, true>(q, x, nullptr, xq, cols, ld, rows, sums, dup);
}

sycl::event iq_mmq_grouped(sycl::queue& q, int t, const void* w, size_t expert_bytes, int64_t w_rows, int64_t w_cols,
                           int n, const void* xq, int64_t total_rows, const int32_t* bounds, int64_t max_rows,
                           const int32_t* ids, float* dst, int64_t ld_dst, const uint8_t* const* wtab, size_t woff,
                           int64_t rows) {
#define STRATA_MMQ_CASE(TY)                                                                                           \
    case TY:                                                                                                          \
        return by_rows<TY>(q, w, wtab, woff, expert_bytes, w_rows, w_cols, n, xq, total_rows, bounds, max_rows, ids, \
                           dst, ld_dst, rows > 0 ? rows : total_rows)
    switch (t) {
        STRATA_MMQ_CASE(6);
        STRATA_MMQ_CASE(8);
        STRATA_MMQ_CASE(10);
        STRATA_MMQ_CASE(11);
        STRATA_MMQ_CASE(12);
        STRATA_MMQ_CASE(13);
        STRATA_MMQ_CASE(14);
        STRATA_MMQ_CASE(kNativeQ6KRows);
        STRATA_MMQ_CASE(kNativeQ8Rows);
        STRATA_MMQ_CASE(16);
        STRATA_MMQ_CASE(17);
        STRATA_MMQ_CASE(18);
        STRATA_MMQ_CASE(20);
        STRATA_MMQ_CASE(21);
        STRATA_MMQ_CASE(22);
        STRATA_MMQ_CASE(23);
        STRATA_MMQ_CASE(29);
        STRATA_MMQ_CASE(42);
        default: return {};
    }
#undef STRATA_MMQ_CASE
}

}  // namespace strata::kernels
