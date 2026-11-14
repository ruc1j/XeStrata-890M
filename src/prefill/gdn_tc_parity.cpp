// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/prefill/gdn_tc_parity.cpp - the matrix-engine DeltaNet scan of a prompt chunk (gdn_rec_tc, STRATA_GDN_TC) against
// the FP32 recurrence kernels (gdn_recurrence_state_only), on the same random inputs: the state after the chunk and the
// output rows, for 16 key heads and 32, 48 and the compiled shape's value heads (core/geometry.hpp: 64 for q4_mid and
// q4_big, whose 4 value heads a key head the prep's A takes over the k and q rows).  The two sum in other orders (the
// scan splits its operands into FP16 pairs), so the check is a relative error, not bits.  GPU, synthetic, no model.
//     gdn_tc_parity [--tokens T]   (default 256)
#include "strata/core/geometry.hpp"
#include "strata/core/gpu.hpp"
#include "strata/core/runtime.hpp"
#include "strata/prefill/kernels.hpp"

#include "gdn_tc.hpp"

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

constexpr int S = 128;

// the largest difference against the reference's rms
double rel_err(const std::vector<float>& a, const std::vector<float>& b) {
    double worst = 0.0, sq = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        worst = std::max(worst, (double) std::fabs(a[i] - b[i]));
        sq += (double) b[i] * b[i];
    }
    const double rms = std::sqrt(sq / (double) b.size());
    return rms > 0.0 ? worst / rms : worst;
}

// -1: no scan for this shape on this GPU (skipped), 0: pass, 1: fail
int check(sycl::queue& q, void* stream, int hk, int hv, int64_t T) {
    const int64_t C = 2 * (int64_t) hk * S + (int64_t) hv * S;
    std::mt19937 rng(7 + hv);
    std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
    std::vector<float> h((size_t) (T * C)), gate((size_t) (T * hv)), beta((size_t) (T * hv)), st0((size_t) S * hv * S);
    for (int64_t t = 0; t < T; ++t) {
        float* ht = h.data() + t * C;
        for (int64_t i = 0; i < C; ++i) ht[i] = uni(rng);
        for (int part = 0; part < 2; ++part)   // q and k heads L2-normalised, as gdn_conv leaves them
            for (int head = 0; head < hk; ++head) {
                float* v = ht + (size_t) part * hk * S + (size_t) head * S;
                double n = 0.0;
                for (int i = 0; i < S; ++i) n += (double) v[i] * v[i];
                const float r = (float) (1.0 / std::sqrt(n + 1e-6));
                for (int i = 0; i < S; ++i) v[i] *= r;
            }
        for (int head = 0; head < hv; ++head) {
            gate[(size_t) (t * hv + head)] = -0.02f - 0.3f * (0.5f + 0.5f * uni(rng));   // log decay, as gdn_gates writes
            beta[(size_t) (t * hv + head)] = 0.5f + 0.45f * uni(rng);
        }
    }
    for (auto& x : st0) x = 0.1f * uni(rng);
    const size_t nh = h.size(), ng = gate.size(), ns = st0.size(), ny = (size_t) T * hv * S;
    float *dh = sycl::malloc_device<float>(nh, q), *dg = sycl::malloc_device<float>(ng, q), *db = sycl::malloc_device<float>(ng, q);
    float *s_ref = sycl::malloc_device<float>(ns, q), *s_tc = sycl::malloc_device<float>(ns, q);
    float *y_ref = sycl::malloc_device<float>(ny, q), *y_tc = sycl::malloc_device<float>(ny, q);
    q.memcpy(dh, h.data(), nh * 4).wait();
    q.memcpy(dg, gate.data(), ng * 4).wait();
    q.memcpy(db, beta.data(), ng * 4).wait();
    q.memcpy(s_ref, st0.data(), ns * 4).wait();
    q.memcpy(s_tc, st0.data(), ns * 4).wait();
    strata::prefill::gdn_recurrence_state_only(s_ref, dh, dg, db, y_ref, T, stream, hk, hv);
    q.wait_and_throw();
    const bool ran = strata::prefill::gdn_rec_tc(q, hk, hv, s_tc, dh, dg, db, y_tc, T);
    q.wait_and_throw();
    int rc = -1;
    if (ran) {
        std::vector<float> a(ns), b(ns), ya(ny), yb(ny);
        q.memcpy(a.data(), s_tc, ns * 4).wait();
        q.memcpy(b.data(), s_ref, ns * 4).wait();
        q.memcpy(ya.data(), y_tc, ny * 4).wait();
        q.memcpy(yb.data(), y_ref, ny * 4).wait();
        const double es = rel_err(a, b), ey = rel_err(ya, yb);
        // the scan's FP16 pairs keep about 22 bits: 1e-3 of the rms leaves room for the order, not for a wrong head
        rc = es < 1e-3 && ey < 1e-3 && std::isfinite(es) && std::isfinite(ey) ? 0 : 1;
        std::printf("  %2d key x %2d value heads, %lld tokens: state %.3g, output %.3g of the rms  %s\n", hk, hv,
                    (long long) T, es, ey, rc == 0 ? "pass" : "FAIL");
    } else {
        std::printf("  %2d key x %2d value heads: no scan for this shape on this GPU (skipped)\n", hk, hv);
    }
    for (float* p : {dh, dg, db, s_ref, s_tc, y_ref, y_tc}) sycl::free(p, q);
    return rc;
}

}  // namespace

int main(int argc, char** argv) {
    int64_t T = 256;
    for (int i = 1; i < argc; ++i)
        if (!std::strcmp(argv[i], "--tokens") && i + 1 < argc) T = std::strtoll(argv[++i], nullptr, 10);
    // the reference: the FP32 kernels (not the chunked WMMA recurrence, not the scan itself)
#if defined(_WIN32)
    _putenv_s("STRATA_GDN_CHUNKED", "0");
#else
    setenv("STRATA_GDN_CHUNKED", "0", 1);
    unsetenv("STRATA_GDN_TC");
#endif
    void* stream = strata::gpu::stream_create();
    sycl::queue& q = strata::core::Runtime::get().stream(stream);
    std::printf("gdn_tc_parity (compiled shape %s)\n", strata::core::kGeometryName);
    const int chk = (int) strata::core::kCompiledGeometry.ssm_k_heads, chv = (int) strata::core::kCompiledGeometry.ssm_v_heads;
    int fails = 0, runs = 0;
    std::vector<std::pair<int, int>> shapes = {{16, 32}, {16, 48}};
    if (!(chk == 16 && (chv == 32 || chv == 48))) shapes.push_back({chk, chv});
    for (const auto& [hk, hv] : shapes) {
        const int r = check(q, stream, hk, hv, T);
        if (r >= 0) ++runs;
        if (r > 0) ++fails;
    }
    std::printf(fails ? "gdn_tc_parity: %d of %d FAILED\n" : "gdn_tc_parity: %d of %d failed\n", fails, runs);
    return fails ? 1 : 0;
}
