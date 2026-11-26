// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/mtp_glue_parity.cpp - the launches STRATA_MTP_GLUE_FUSE takes out of the MTP layer and the verify
// window's tail, each against the launches it replaces, memcmp.
//
//  - the RMS norm that writes its q8_1 image (plain, at the embedding's width and at the hidden input's hc * n_embd,
//    and grouped) against the norm and native_quantize_q8_1 of its output: the output and the image;
//  - the final mixer's read with the residual write folded in (apply, no injection weight) against gr_write_multi and
//    the read without it: R and mixed, with the BF16 projections and with the Q8_0 ones, T = 1..8.
// Real geometry (n_embd 2560, hc 4, rank 320).  GPU, synthetic, no model.
#include "strata/core/gpu.hpp"
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/gr.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_qsa.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

namespace k = strata::kernels;

namespace {

constexpr int N = 2560, HC = 4, D = N * HC, LR = 320;
int g_cases = 0, g_bad = 0;

template <class T> T* dev(size_t count) {
    T* d = nullptr;
    if (!strata::gpu::alloc_device(&d, count * sizeof(T) + 64)) throw std::runtime_error(strata::gpu::last_error());
    return d;
}
template <class T> T* up(const std::vector<T>& v) {
    T* d = dev<T>(v.size());
    if (!strata::gpu::copy(d, v.data(), v.size() * sizeof(T))) throw std::runtime_error(strata::gpu::last_error());
    return d;
}
template <class T> std::vector<T> down(const T* d, size_t count) {
    std::vector<T> h(count);
    if (!strata::gpu::copy(h.data(), d, count * sizeof(T))) throw std::runtime_error(strata::gpu::last_error());
    return h;
}
template <class T> bool same(const std::vector<T>& a, const std::vector<T>& b) {
    return a.size() == b.size() &&
           std::memcmp(static_cast<const void*>(a.data()), static_cast<const void*>(b.data()), a.size() * sizeof(T)) == 0;
}
void verdict(bool ok, const char* what, int a, int b) {
    ++g_cases;
    if (ok) return;
    ++g_bad;
    std::printf("    *** %s (%d, %d) differs ***\n", what, a, b);
}

uint16_t f16_bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    const uint32_t s = (u >> 16u) & 0x8000u, e = ((u >> 23u) & 0xffu) - 127u + 15u, m = (u >> 13u) & 0x3ffu;
    return (uint16_t) (s | (e << 10u) | m);
}
// one matrix in kNativeQ8Rows' layout: each row's int8 values, then its fp16 scales
std::vector<uint8_t> q8_rows(int rows, int cols, std::mt19937& rng) {
    const size_t row = (size_t) cols / 32 * 34;
    std::vector<uint8_t> m((size_t) rows * row);
    for (int r = 0; r < rows; ++r) {
        uint8_t* p = m.data() + (size_t) r * row;
        for (int c = 0; c < cols; ++c) p[c] = (uint8_t) (int8_t) ((int) (rng() % 255) - 127);
        for (int b = 0; b < cols / 32; ++b) {
            const uint16_t d = f16_bits(0.0001f + 0.00001f * (float) (rng() % 50));
            std::memcpy(p + (size_t) cols + 2 * (size_t) b, &d, 2);
        }
    }
    return m;
}

// the norm with its q8_1 image against the norm and the quantization: n_rows rows of n_cols, gamma rows `groups`
// (0: the plain norm), the image as n_rows * n_cols / n_in rows of n_in
void norm_case(int n_cols, int n_rows, int groups, int n_in, std::mt19937& rng, void* s) {
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    std::vector<float> x((size_t) n_cols * n_rows), gamma((size_t) n_cols * (groups ? groups : 1));
    for (auto& v : x) v = gauss(rng) * 3.0f;
    for (int c = 0; c < n_cols && n_rows > 1; ++c) x[(size_t) n_cols + c] = 0.0f;   // a row of zeros
    for (auto& v : gamma) v = 1.0f + 0.2f * gauss(rng);
    const size_t q8 = x.size() / 32 * 36;
    float *d_x = up(x), *d_g = up(gamma), *d_a = dev<float>(x.size()), *d_b = dev<float>(x.size());
    uint8_t *q_a = dev<uint8_t>(q8), *q_b = dev<uint8_t>(q8);
    const float eps = 1e-6f;
    if (groups) {
        k::native_qsa_rms_norm_grouped_q8_1(d_x, d_g, d_a, q_a, n_cols, groups, n_rows, eps, s);
        k::native_qsa_rms_norm_grouped(d_x, d_g, d_b, n_cols, groups, n_rows, eps, s);
    } else {
        k::native_qsa_rms_norm_weighted_q8_1(d_x, d_g, d_a, q_a, n_cols, n_rows, eps, s);
        k::native_qsa_rms_norm_weighted(d_x, d_g, d_b, n_cols, n_rows, eps, s);
    }
    k::native_quantize_q8_1(d_b, q_b, n_in, (int) (x.size() / (size_t) n_in), s);
    strata::gpu::stream_sync(s);
    verdict(same(down(d_a, x.size()), down(d_b, x.size())), groups ? "the grouped norm's output" : "the norm's output",
            n_cols, n_rows);
    verdict(same(down(q_a, q8), down(q_b, q8)), groups ? "the grouped norm's q8_1 image" : "the norm's q8_1 image",
            n_cols, n_rows);
    for (void* p : {(void*) d_x, (void*) d_g, (void*) d_a, (void*) d_b, (void*) q_a, (void*) q_b}) strata::gpu::free(p);
}

// the final mixer's read with the write folded in against the write and then the read
void read_case(int T, bool q8, std::mt19937& rng, void* s) {
    std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
    std::vector<uint16_t> bf((size_t) LR * D);
    for (auto& x : bf) x = (uint16_t) (0x3c00u | (rng() & 0x7fu));   // small BF16 values
    std::vector<float> norm(D), R((size_t) T * D), bo((size_t) T * N), inj((size_t) T * HC);
    for (auto& x : norm) x = 1.0f + 0.1f * uni(rng);
    for (auto& x : R) x = uni(rng);
    for (auto& x : bo) x = 0.1f * uni(rng);
    for (auto& x : inj) x = 3.0f * uni(rng);
    uint16_t *wd = up(bf), *wu = up(bf);
    uint8_t *qd = q8 ? up(q8_rows(LR, D, rng)) : nullptr, *qu = q8 ? up(q8_rows(D, LR, rng)) : nullptr;
    float *wn = up(norm), *d_bo = up(bo), *d_inj = up(inj);
    float *lo = dev<float>((size_t) T * LR), *rs = dev<float>((size_t) T * HC);
    float* xn = dev<float>((size_t) k::kFusedGrMaxT * D);
    std::vector<float> got_R[2], got_m[2];
    for (int fold = 0; fold < 2; ++fold) {
        float *d_R = up(R), *mixed = dev<float>((size_t) T * N);
        std::vector<k::FusedGrArgs> a(T);
        for (int t = 0; t < T; ++t) {
            auto& x = a[t];
            x.R = d_R + (size_t) t * D; x.R_out = d_R + (size_t) t * D; x.apply = fold != 0;
            if (fold) { x.bo_prev = d_bo + (size_t) t * N; x.inj_prev = d_inj + (size_t) t * HC; }
            x.w_norm = wn; x.w_down = wd; x.w_up = wu; x.q8_down = qd; x.q8_up = qu;
            x.lo = lo + (size_t) t * LR; x.rs = rs + (size_t) t * HC; x.mixed = mixed + (size_t) t * N;
        }
        if (!fold) k::gr_write_multi(d_R, d_bo, d_inj, k::GrShapes{N, HC, LR}, d_R, T, s);
        k::fused_gr_read_multi(a.data(), T, xn, s);
        strata::gpu::stream_sync(s);
        got_R[fold] = down(d_R, (size_t) T * D);
        got_m[fold] = down(mixed, (size_t) T * N);
        strata::gpu::free(d_R);
        strata::gpu::free(mixed);
    }
    verdict(same(got_R[0], got_R[1]), q8 ? "the folded read's R (Q8_0)" : "the folded read's R (BF16)", T, 0);
    verdict(same(got_m[0], got_m[1]), q8 ? "the folded read's mixed (Q8_0)" : "the folded read's mixed (BF16)", T, 0);
    for (void* p : {(void*) wd, (void*) wu, (void*) qd, (void*) qu, (void*) wn, (void*) d_bo, (void*) d_inj, (void*) lo,
                    (void*) rs, (void*) xn})
        if (p) strata::gpu::free(p);
}

}  // namespace

int main() try {
    std::mt19937 rng(52);  // NOLINT(bugprone-random-generator-seed): reproducible rows.
    void* s = strata::gpu::stream_create();
    if (s == nullptr) { std::printf("mtp_glue_parity: no stream: %s\n", strata::gpu::last_error()); return 1; }
    for (int rows = 1; rows <= 8; ++rows) norm_case(N, rows, 0, N, rng, s);     // the embedding's norm
    for (int rows = 1; rows <= 2; ++rows) norm_case(D, rows, 0, N, rng, s);     // the hidden input's, one RMS a row
    for (int rows = HC; rows <= 2 * HC; rows += HC) norm_case(N, rows, HC, N, rng, s);   // one RMS a stream
    if (k::fused_gr_supported(N, HC, LR))
        for (int q8 = 0; q8 < 2; ++q8)
            for (int T = 1; T <= k::kFusedGrMaxT; ++T) read_case(T, q8 != 0, rng, s);
    else std::printf("  (no fused read on this device: the folded read is not checked)\n");
    strata::gpu::stream_destroy(s);
    std::printf("mtp_glue: %d cases, %d failures\n", g_cases, g_bad);
    if (g_bad) return 1;
    std::printf("mtp_glue_parity OK\n");
    return 0;
} catch (const std::exception& e) {
    std::printf("mtp_glue_parity: %s\n", e.what());
    return 1;
}
