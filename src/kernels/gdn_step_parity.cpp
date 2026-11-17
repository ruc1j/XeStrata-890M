// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/gdn_step_parity.cpp - the decode window's GDN step at the model's shape (state 128, 16 key heads, 48
// value heads): the recurrence over the window's rows and the output norm.
//
//   * every form of gdn_step_norm_multi (one group a head, split, prefetch), windows of 1..8 rows: the outputs
//     of the verify launch and the state a commit of every row leaves, against a host reference in FP64 (relative
//     1e-4: the kernels sum 128 products in FP32 by row groups), the verify launch leaving the state as it was;
//   * the forms against the split form, memcmp (the header says they give the same bits);
//   * a commit that keeps fewer rows than the window has (the count read from device memory), the same way;
//   * gdn_step_norm_pending (the split form): 0..12 pending rows run before the window's, the state written back at
//     write_at 1 and 6 rows - against the reference; its q8_1 image (the next projection's input, written by the
//     same launch) against the image of the outputs it wrote.
#include "strata/core/gpu.hpp"
#include "strata/kernels/gdn.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

constexpr int S = 128, HK = 16, HV = 48, QK = S * HK, C = 2 * QK + S * HV, MT = 8, PEND = 12;
constexpr float kEps = 1e-6f;
constexpr size_t kState = (size_t) S * S * HV;

struct Rows {   // q|k|v rows (n, C), gate and beta (n, HV)
    const float* h;
    const float* gate;
    const float* beta;
};

// The recurrence over rows [0, n) of `r` from `state` (in place); with `y`, the unnormalized outputs of every row
void ref_rows(std::vector<double>& state, const Rows& r, int n, std::vector<double>* y) {
    std::vector<double> k(S), q(S);
    for (int t = 0; t < n; ++t)
        for (int head = 0; head < HV; ++head) {
            const int qh = head % HK;
            const double g = std::exp((double) r.gate[(size_t) t * HV + head]);
            const double b = (double) r.beta[(size_t) t * HV + head];
            const float* ht = r.h + (size_t) t * C;
            for (int row = 0; row < S; ++row) {
                k[(size_t) row] = ht[QK + qh * S + row];
                q[(size_t) row] = ht[qh * S + row];
            }
            for (int col = 0; col < S; ++col) {
                double kv = 0.0;
                for (int row = 0; row < S; ++row) kv += state[((size_t) row * HV + head) * S + col] * k[(size_t) row];
                const double delta = ((double) ht[2 * QK + head * S + col] - g * kv) * b;
                double o = 0.0;
                for (int row = 0; row < S; ++row) {
                    double& sv = state[((size_t) row * HV + head) * S + col];
                    sv = g * sv + k[(size_t) row] * delta;
                    o += sv * q[(size_t) row];
                }
                if (y != nullptr) (*y)[(size_t) t * S * HV + (size_t) (head * S + col)] = o / std::sqrt((double) S);
            }
        }
}

// y = oc * rsqrt(mean(oc^2) + eps) * gamma * sigmoid(z) a head and row (or z * sigmoid(z): gdn_silu_gate)
void ref_norm(std::vector<double>& y, const float* z, const float* gamma, int n) {
    const bool silu = strata::kernels::gdn_silu_gate();
    for (int t = 0; t < n; ++t)
        for (int head = 0; head < HV; ++head) {
            double ss = 0.0;
            for (int col = 0; col < S; ++col) {
                const double oc = y[(size_t) t * S * HV + (size_t) (head * S + col)];
                ss += oc * oc;
            }
            const double scale = 1.0 / std::sqrt(ss / S + (double) kEps);
            for (int col = 0; col < S; ++col) {
                const size_t at = (size_t) t * S * HV + (size_t) (head * S + col);
                const double zz = z[at], gz = 1.0 / (1.0 + std::exp(-zz));
                y[at] = y[at] * scale * (double) gamma[col] * (silu ? zz * gz : gz);
            }
        }
}

bool close(const std::vector<float>& got, const std::vector<double>& want, size_t n, const char* what) {
    for (size_t i = 0; i < n; ++i) {
        const double d = std::fabs((double) got[i] - want[i]);
        if (!(d <= 1e-4 * std::fabs(want[i]) + 1e-6)) {
            std::printf("    *** %s: value %zu is %g, the reference %g ***\n", what, i, (double) got[i], want[i]);
            return false;
        }
    }
    return true;
}

bool same(const std::vector<float>& a, const std::vector<float>& b, size_t n) {
    return std::memcmp(a.data(), b.data(), n * sizeof(float)) == 0;
}

}  // namespace

int main() {
    using namespace strata::kernels;
    std::mt19937 rng(5);  // NOLINT(bugprone-random-generator-seed): reproducible fixtures.
    std::uniform_real_distribution<float> u(-0.05f, 0.05f);
    auto fill = [&](size_t n) { std::vector<float> v(n); for (auto& x : v) x = u(rng); return v; };
    const auto st = fill(kState), h = fill((size_t) MT * C), z = fill((size_t) MT * S * HV), gm = fill(S);
    const auto ph = fill((size_t) PEND * C);
    std::vector<float> gate((size_t) MT * HV), beta((size_t) MT * HV), pg((size_t) PEND * HV), pb((size_t) PEND * HV);
    for (auto& x : gate) x = -0.1f + u(rng);
    for (auto& x : beta) x = 0.5f + u(rng);
    for (auto& x : pg) x = -0.1f + u(rng);
    for (auto& x : pb) x = 0.5f + u(rng);
    float *d_st, *d_h, *d_g, *d_b, *d_z, *d_gm, *d_y, *d_ph, *d_pg, *d_pb;
    int32_t *d_keep, *d_pn;
    uint8_t* d_q8;
    const size_t kQ8 = (size_t) MT * S * HV / 32 * 36;   // the window's q8_1 image: 36 bytes a 32-value block
    if (!strata::gpu::alloc_device(&d_st, kState * 4) || !strata::gpu::alloc_device(&d_h, h.size() * 4) ||
        !strata::gpu::alloc_device(&d_g, gate.size() * 4) || !strata::gpu::alloc_device(&d_b, beta.size() * 4) ||
        !strata::gpu::alloc_device(&d_z, z.size() * 4) || !strata::gpu::alloc_device(&d_gm, gm.size() * 4) ||
        !strata::gpu::alloc_device(&d_y, z.size() * 4) || !strata::gpu::alloc_device(&d_ph, ph.size() * 4) ||
        !strata::gpu::alloc_device(&d_pg, pg.size() * 4) || !strata::gpu::alloc_device(&d_pb, pb.size() * 4) ||
        !strata::gpu::alloc_device(&d_keep, 4) || !strata::gpu::alloc_device(&d_pn, 4) ||
        !strata::gpu::alloc_device(&d_q8, kQ8)) {
        std::printf("gdn_step_parity: device allocation failed\n");
        return 1;
    }
    strata::gpu::copy(d_h, h.data(), h.size() * 4);
    strata::gpu::copy(d_g, gate.data(), gate.size() * 4);
    strata::gpu::copy(d_b, beta.data(), beta.size() * 4);
    strata::gpu::copy(d_z, z.data(), z.size() * 4);
    strata::gpu::copy(d_gm, gm.data(), gm.size() * 4);
    strata::gpu::copy(d_ph, ph.data(), ph.size() * 4);
    strata::gpu::copy(d_pg, pg.data(), pg.size() * 4);
    strata::gpu::copy(d_pb, pb.data(), pb.size() * 4);
    void* s = strata::gpu::stream_create();
    const std::vector<double> st64(st.begin(), st.end());
    const Rows win{h.data(), gate.data(), beta.data()}, pend{ph.data(), pg.data(), pb.data()};
    const char* names[] = {"one group a head", "split", "prefetch"};
    int bad = 0;

    // ---- the forms of gdn_step_norm_multi
    std::vector<std::vector<float>> y_split((size_t) MT + 1), s_split((size_t) MT + 1), p_split((size_t) MT + 1);
    for (const int form : {1, 0, 2}) {   // the split first: the others are compared with it
        gdn_step_force_form(form);
        int form_bad = 0;
        for (int T = 1; T <= MT; ++T) {
            const size_t ny = (size_t) T * S * HV;
            std::vector<double> ref_s = st64, ref_y(ny);
            ref_rows(ref_s, win, T, &ref_y);
            ref_norm(ref_y, z.data(), gm.data(), T);
            std::vector<float> y(ny), kept(kState), after(kState);
            const int32_t keep = T;
            strata::gpu::copy(d_keep, &keep, 4);
            strata::gpu::copy(d_st, st.data(), kState * 4);
            gdn_step_norm_multi(d_st, d_h, C, d_g, d_b, d_z, d_gm, kEps, d_y, HK, HV, T, nullptr, s);   // verify
            strata::gpu::stream_sync(s);
            strata::gpu::copy(y.data(), d_y, ny * 4);
            strata::gpu::copy(kept.data(), d_st, kState * 4);
            gdn_step_norm_multi(d_st, d_h, C, d_g, d_b, d_z, d_gm, kEps, d_y, HK, HV, T, d_keep, s, T);   // commit
            strata::gpu::stream_sync(s);
            strata::gpu::copy(after.data(), d_st, kState * 4);
            bool ok = close(y, ref_y, ny, "outputs") && close(after, ref_s, kState, "committed state");
            if (!same(kept, st, kState)) { std::printf("    *** the verify launch wrote the state ***\n"); ok = false; }
            // a commit of fewer rows than the window has
            const int32_t part = T / 2;
            std::vector<double> ref_p = st64;
            ref_rows(ref_p, win, part, nullptr);
            std::vector<float> after_p(kState);
            strata::gpu::copy(d_keep, &part, 4);
            strata::gpu::copy(d_st, st.data(), kState * 4);
            gdn_step_norm_multi(d_st, d_h, C, d_g, d_b, d_z, d_gm, kEps, d_y, HK, HV, T, d_keep, s, T);
            strata::gpu::stream_sync(s);
            strata::gpu::copy(after_p.data(), d_st, kState * 4);
            ok = close(after_p, ref_p, kState, "the state after a commit of some rows") && ok;
            if (form == 1) {
                y_split[(size_t) T] = y;
                s_split[(size_t) T] = after;
                p_split[(size_t) T] = after_p;
            } else if (!same(y, y_split[(size_t) T], ny) || !same(after, s_split[(size_t) T], kState) ||
                       !same(after_p, p_split[(size_t) T], kState)) {
                std::printf("    *** not the split form's bits ***\n");
                ok = false;
            }
            if (!ok) { std::printf("    (form %d, %s, %d rows)\n", form, names[form], T); ++form_bad; }
        }
        std::printf("  %-18s windows of 1..%d rows  %s\n", names[form], MT,
                    form_bad ? "*** NO ***" : form == 1 ? "matches the reference" : "matches the reference, the split's bits");
        bad += form_bad;
    }

    // ---- the pending rows first (gdn_step_norm_pending)
    int pend_bad = 0, cases = 0;
    for (const int np : {0, 3, 7, PEND})
        for (const int write_at : {1, 6})
            for (const int T : {1, 3, MT}) {
                const size_t ny = (size_t) T * S * HV;
                std::vector<double> ref_s = st64, ref_y(ny);
                ref_rows(ref_s, pend, np, nullptr);
                const std::vector<double> ref_written = np >= write_at ? ref_s : st64;   // below write_at: as read
                ref_rows(ref_s, win, T, &ref_y);
                ref_norm(ref_y, z.data(), gm.data(), T);
                const size_t nq = ny / 32 * 36;
                {
                    const int form = 1;
                    gdn_step_force_form(form);
                    const int32_t n32 = np;
                    strata::gpu::copy(d_pn, &n32, 4);
                    strata::gpu::copy(d_st, st.data(), kState * 4);
                    const GdnPending pd{d_ph, d_pg, d_pb, d_pn, write_at};
                    gdn_step_norm_pending(d_st, d_h, C, d_g, d_b, d_z, d_gm, kEps, d_y, HK, HV, T, nullptr, s, 0, d_q8, pd);
                    strata::gpu::stream_sync(s);
                    std::vector<float> y(ny), after(kState);
                    std::vector<uint8_t> q8(nq), q8_ref(nq);
                    strata::gpu::copy(y.data(), d_y, ny * 4);
                    strata::gpu::copy(after.data(), d_st, kState * 4);
                    strata::gpu::copy(q8.data(), d_q8, nq);
                    native_quantize_q8_1(d_y, d_q8, S * HV, T, s);   // the image of the outputs, by the kernel of its own
                    strata::gpu::stream_sync(s);
                    strata::gpu::copy(q8_ref.data(), d_q8, nq);
                    bool ok = close(y, ref_y, ny, "outputs") && close(after, ref_written, kState, "state");
                    if (std::memcmp(q8.data(), q8_ref.data(), nq) != 0) {
                        std::printf("    *** the q8_1 image is not the outputs' ***\n");
                        ok = false;
                    }
                    ++cases;
                    if (!ok) {
                        std::printf("    (%s, %d pending rows, written at %d, %d rows)\n", names[form], np, write_at, T);
                        ++pend_bad;
                    }
                }
            }
    std::printf("  %-18s %d cases  %s\n", "pending rows", cases, pend_bad ? "*** NO ***" : "match the reference, the q8_1 image the outputs'");
    bad += pend_bad;
    gdn_step_force_form(-1);

    std::printf("\ngdn_step: %d failures\n", bad);
    if (bad) return 1;
    std::printf("gdn_step_parity OK\n");
    return 0;
}
