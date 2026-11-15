// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/prefill/gdn_prompt_parity.cpp - the prompt path's DeltaNet kernels (prefill/kernels.hpp: gdn_gates, gdn_conv,
// gdn_recurrence, gdn_recurrence_state_only; gdn_tc.hpp: gdn_rec_tc) against a CPU reference in double, on the same
// random inputs, for every shape the kernels are built for: the compiled one (core/geometry.hpp) and the plain models'.
//
// The reference is the layer as kernels/gdn.hpp states it (ref_step of kernels/gdn_parity.cpp, a token after the
// other): the head pairing h % h_k, the decay before the rank-1 update, the state as (S, h_v, S).  Two chunks are
// read one after the other, as the engine reads a prompt: the conv history and the state go from the first into the
// second.  The first is long enough for every form of the recurrence (the chunked one from 64 tokens, the scan from
// 128, more than one of the scan's super-blocks) and is no multiple of their chunks; the second is shorter than all
// of them.  Per chunk:
//   gates, conv      the gate and beta of every token, the conv's output and the history it leaves
//   FP32             gdn_recurrence_state_only: the state and each token's output before the norm
//   with the output  gdn_recurrence: the state and the normalized FP16 output.  Where the device takes the chunked
//                    recurrence (STRATA_GDN_CHUNKED, the default) it is that one, with FP16 operands; elsewhere the
//                    FP32 kernels and the norm kernel.  Which ran is told by the state's bits against the FP32 run's.
//   the scan         gdn_rec_tc (STRATA_GDN_TC), called as Gdn::recurrence does from 128 tokens; a GPU without it skips
// GPU, synthetic, no model.
//     gdn_prompt_parity [--tokens T] [--chunked 0] [--keyhead 0]
//       --tokens T    the first chunk's tokens (default 2100)
//       --chunked 0   STRATA_GDN_CHUNKED=0: the run with the output takes the FP32 kernels
//       --keyhead 0   STRATA_GDN_KEYHEAD=0: the FP32 kernels are D-2, not the faster of D-2 and the key-head form
#include "strata/core/geometry.hpp"
#include "strata/core/gpu.hpp"
#include "strata/core/runtime.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/prefill/kernels.hpp"

#include "gdn_tc.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <utility>
#include <vector>

namespace {

constexpr int S = 128;
constexpr float EPS = 1e-6f;
constexpr int64_t kScanMin = 128;   // kernels.cpp's kGdnTcMin: Gdn::recurrence takes the scan from this many tokens

void set_env(const char* name, const char* value) {   // nullptr: unset
#if defined(_WIN32)
    _putenv_s(name, value != nullptr ? value : "");
#else
    if (value != nullptr) setenv(name, value, 1);
    else unsetenv(name);
#endif
}

// the largest difference against the reference's rms; a value that is not finite fails whatever the others are
struct Err {
    double worst = 0.0;
    bool finite = true;
    bool within(double tol) const { return finite && worst <= tol; }
};
template <class Got> Err rel_err(const std::vector<Got>& got, const std::vector<double>& want) {
    Err e;
    double sq = 0.0;
    for (size_t i = 0; i < want.size(); ++i) {
        const double g = (double) got[i];
        if (!std::isfinite(g)) e.finite = false;
        else e.worst = std::max(e.worst, std::fabs(g - want[i]));
        sq += want[i] * want[i];
    }
    const double rms = std::sqrt(sq / (double) std::max<size_t>(want.size(), 1));
    if (rms > 0.0) e.worst /= rms;
    return e;
}

double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

// The reference of one layer: the state (S, h_v, S) with the column fastest and the conv history [C][3], both as the
// kernels hold them, carried from chunk to chunk.
struct Ref {
    int hk, hv, C;
    std::vector<double> state, hist;
    std::vector<double> gate, beta, h, o, y;   // the last chunk's: [T][hv], [T][hv], [T][C], [T][hv][S], [T][hv][S]

    Ref(int k, int v) : hk(k), hv(v), C(2 * k * S + v * S) {}

    void gates(const std::vector<float>& ab, const std::vector<float>& dt, const std::vector<float>& a, int64_t T) {
        gate.resize((size_t) (T * hv));
        beta.resize((size_t) (T * hv));
        for (int64_t t = 0; t < T; ++t)
            for (int hd = 0; hd < hv; ++hd) {
                const double v = (double) ab[(size_t) (t * 2 * hv + hd)] + (double) dt[(size_t) hd];
                gate[(size_t) (t * hv + hd)] = std::log1p(std::exp(v)) * (double) a[(size_t) hd];
                beta[(size_t) (t * hv + hd)] = sigmoid((double) ab[(size_t) (t * 2 * hv + hv + hd)]);
            }
    }

    // the 4-tap causal conv and SiLU, then the L2 norm of every token's q and k heads
    void conv(const std::vector<float>& qkv, const std::vector<float>& w, int64_t T) {
        h.resize((size_t) (T * C));
        for (int c = 0; c < C; ++c) {
            double v0 = hist[(size_t) c * 3], v1 = hist[(size_t) c * 3 + 1], v2 = hist[(size_t) c * 3 + 2];
            const double w0 = w[(size_t) c * 4], w1 = w[(size_t) c * 4 + 1], w2 = w[(size_t) c * 4 + 2], w3 = w[(size_t) c * 4 + 3];
            for (int64_t t = 0; t < T; ++t) {
                const double x = qkv[(size_t) (t * C + c)];
                const double s = v0 * w0 + v1 * w1 + v2 * w2 + x * w3;
                h[(size_t) (t * C + c)] = s * sigmoid(s);
                v0 = v1; v1 = v2; v2 = x;
            }
            hist[(size_t) c * 3] = v0; hist[(size_t) c * 3 + 1] = v1; hist[(size_t) c * 3 + 2] = v2;
        }
        for (int64_t t = 0; t < T; ++t)
            for (int hd = 0; hd < 2 * hk; ++hd) {
                double* v = h.data() + (size_t) (t * C + (int64_t) hd * S);
                double ss = 0.0;
                for (int i = 0; i < S; ++i) ss += v[i] * v[i];
                const double r = 1.0 / std::sqrt(ss + (double) EPS);
                for (int i = 0; i < S; ++i) v[i] *= r;
            }
    }

    // The recurrence over the chunk from `st`, which it updates; out[t][head][col] the output before the norm.
    // `modulo` false is the rival pairing (h / (h_v / h_k)), for the check that this input can tell the two apart.
    void recurrence(std::vector<double>& st, std::vector<double>& out, int64_t T, bool modulo) const {
        out.assign((size_t) (T * hv * S), 0.0);
        std::atomic<int> next{0};
        auto work = [&] {
            std::vector<double> s((size_t) S * S);   // s[col][row] of one value head
            for (int hd = next++; hd < hv; hd = next++) {
                const int qh = modulo ? hd % hk : hd / (hv / hk);
                for (int i = 0; i < S; ++i)
                    for (int j = 0; j < S; ++j) s[(size_t) j * S + i] = st[((size_t) i * hv + hd) * S + j];
                for (int64_t t = 0; t < T; ++t) {
                    const double* ht = h.data() + (size_t) (t * C);
                    const double *q = ht + (size_t) qh * S, *k = ht + (size_t) (hk + qh) * S, *v = ht + (size_t) (2 * hk + hd) * S;
                    const double g = std::exp(gate[(size_t) (t * hv + hd)]), b = beta[(size_t) (t * hv + hd)];
                    for (int j = 0; j < S; ++j) {
                        double* sj = s.data() + (size_t) j * S;
                        double kv = 0.0;
                        for (int i = 0; i < S; ++i) kv += sj[i] * k[i];
                        const double d = (v[j] - g * kv) * b;   // the decay first, then the rank-1 update
                        double dot = 0.0;
                        for (int i = 0; i < S; ++i) {
                            sj[i] = g * sj[i] + k[i] * d;
                            dot += sj[i] * q[i];
                        }
                        out[(size_t) ((t * hv + hd) * S + j)] = dot / std::sqrt((double) S);
                    }
                }
                for (int i = 0; i < S; ++i)
                    for (int j = 0; j < S; ++j) st[((size_t) i * hv + hd) * S + j] = s[(size_t) j * S + i];
            }
        };
        std::vector<std::thread> pool;
        const unsigned n = std::max(1u, std::min<unsigned>(std::thread::hardware_concurrency(), (unsigned) hv));
        for (unsigned i = 1; i < n; ++i) pool.emplace_back(work);
        work();
        for (auto& th : pool) th.join();
    }

    // each head's RMS norm, the weight and the output gate: sigmoid(z), or silu(z) (the plain models')
    void norm(const std::vector<float>& z, const std::vector<float>& gamma, int64_t T, bool silu) {
        y.resize(o.size());
        for (int64_t r = 0; r < T * hv; ++r) {
            const double* oc = o.data() + (size_t) r * S;
            double ss = 0.0;
            for (int j = 0; j < S; ++j) ss += oc[j] * oc[j];
            const double inv = 1.0 / std::sqrt(ss / (double) S + (double) EPS);
            for (int j = 0; j < S; ++j) {
                const double zz = (double) z[(size_t) r * S + j];
                y[(size_t) r * S + j] = oc[j] * inv * (double) gamma[(size_t) j] * (silu ? zz * sigmoid(zz) : sigmoid(zz));
            }
        }
    }
};

// a device buffer with a host image's content
template <class T> struct Dev {
    sycl::queue& q;
    size_t n;
    T* p;
    Dev(sycl::queue& queue, size_t count) : q(queue), n(count), p(sycl::malloc_device<T>(count, queue)) {}
    Dev(sycl::queue& queue, const std::vector<T>& host) : Dev(queue, host.size()) { put(host); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
    ~Dev() { sycl::free(p, q); }
    void put(const std::vector<T>& host) { q.memcpy(p, host.data(), n * sizeof(T)).wait(); }
    std::vector<T> get() {
        std::vector<T> host(n);
        q.memcpy(host.data(), p, n * sizeof(T)).wait();
        return host;
    }
};

struct Tally {
    int fails = 0, runs = 0, skipped = 0;
    void line(const char* what, const char* detail, bool ok) {
        ++runs;
        if (!ok) ++fails;
        std::printf("    %-46s %-50s %s\n", what, detail, ok ? "ok" : "FAIL");
    }
};

// Tolerances, as the largest difference over the reference's rms.  Measured on the Radeon 890M (gfx1150, the HIP
// backend) for 16 x 32, 16 x 48 and 16 x 64 heads at 2100 and 45 tokens; they depend on the arithmetic, not on the
// device: a wrong head, a wrong pairing or a state row out of place is of the rms's own size (1 and more).
constexpr double kTolFp32 = 1e-4;     // FP32 against double: gates, conv, the FP32 recurrence, the scan's split FP16 pairs
constexpr double kTolF16Out = 2e-2;   // the FP32 kernels' normalized output, rounded to FP16: half an FP16 step of the
                                      // largest outputs (4 to 8) is 0.002, 0.004 of the rms (0.5) - what was measured
constexpr double kTolChunked = 0.1;   // the chunked recurrence: k, q, T, P, D and the state as FP16 operands

void check(sycl::queue& q, void* stream, int hk, int hv, const std::vector<int64_t>& chunks, Tally& tally) {
    namespace pf = strata::prefill;
    const int C = 2 * hk * S + hv * S;
    std::printf("  %d key x %d value heads\n", hk, hv);
    std::mt19937 rng(11 + (unsigned) hv);
    std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
    auto fill = [&](std::vector<float>& v, float scale, float bias = 0.0f) { for (auto& x : v) x = bias + scale * uni(rng); };
    std::vector<float> dt((size_t) hv), a((size_t) hv), w((size_t) C * 4), gamma((size_t) S), st0((size_t) S * hv * S), hist0((size_t) C * 3);
    fill(dt, 1.0f);
    for (auto& x : a) x = -0.02f - 0.14f * (1.0f + uni(rng));   // ssm_a < 0: the gate is a log decay
    fill(w, 0.5f);
    fill(gamma, 0.2f, 1.0f);
    fill(st0, 0.1f);
    fill(hist0, 1.0f);

    Ref ref(hk, hv);
    ref.state.assign(st0.begin(), st0.end());
    ref.hist.assign(hist0.begin(), hist0.end());
    Dev<float> d_dt(q, dt), d_a(q, a), d_w(q, w), d_gamma(q, gamma), d_state(q, st0), d_hist(q, hist0);

    for (const int64_t T : chunks) {
        std::printf("   a chunk of %lld tokens\n", (long long) T);
        std::vector<float> ab((size_t) (T * 2 * hv)), qkv((size_t) (T * C)), z((size_t) (T * hv * S));
        fill(ab, 2.0f);
        fill(qkv, 1.0f);
        fill(z, 2.0f);
        const std::vector<double> ref_in = ref.state;
        ref.gates(ab, dt, a, T);
        ref.conv(qkv, w, T);
        ref.recurrence(ref.state, ref.o, T, true);
        char detail[160];
        if (hv > hk) {   // the input tells the head pairings apart
            std::vector<double> st = ref_in, o_rival;
            ref.recurrence(st, o_rival, T, false);
            const Err e = rel_err(o_rival, ref.o);
            std::snprintf(detail, sizeof detail, "%.3g of the rms apart", e.worst);
            tally.line("the rival head pairing shows", detail, e.worst > 10 * kTolChunked);   // far past every tolerance
        }

        const size_t nth = (size_t) (T * hv), nh = (size_t) (T * C), ny = (size_t) (T * hv * S), ns = st0.size();
        Dev<float> d_ab(q, ab), d_qkv(q, qkv), d_z(q, z), d_gate(q, nth), d_beta(q, nth), d_h(q, nh), d_y(q, ny);
        Dev<uint16_t> d_y16(q, ny);

        pf::gdn_gates(d_ab.p, d_dt.p, d_a.p, d_gate.p, d_beta.p, T, stream, hk, hv);
        pf::gdn_conv(d_hist.p, d_qkv.p, d_w.p, d_h.p, T, EPS, stream, hk, hv);
        q.wait_and_throw();
        {
            const Err eg = rel_err(d_gate.get(), ref.gate), eb = rel_err(d_beta.get(), ref.beta);
            std::snprintf(detail, sizeof detail, "gate %.3g, beta %.3g", eg.worst, eb.worst);
            tally.line("gates", detail, eg.within(kTolFp32) && eb.within(kTolFp32));
            const Err eh = rel_err(d_h.get(), ref.h), ehist = rel_err(d_hist.get(), ref.hist);
            std::snprintf(detail, sizeof detail, "output %.3g, history %.3g", eh.worst, ehist.worst);
            tally.line("conv", detail, eh.within(kTolFp32) && ehist.within(0.0));   // the history is the inputs themselves
        }

        const std::vector<float> st_in = d_state.get();
        // FP32: in place on the carried state
        pf::gdn_recurrence_state_only(d_state.p, d_h.p, d_gate.p, d_beta.p, d_y.p, T, stream, hk, hv);
        q.wait_and_throw();
        const std::vector<float> st_fp32 = d_state.get();
        {
            const Err es = rel_err(st_fp32, ref.state), eo = rel_err(d_y.get(), ref.o);
            std::snprintf(detail, sizeof detail, "state %.3g, output %.3g", es.worst, eo.worst);
            tally.line("FP32 recurrence", detail, es.within(kTolFp32) && eo.within(kTolFp32));
        }
        // with the output, under each output gate: the chunked recurrence where the device takes it and the chunk is
        // long enough
        for (const bool silu : {false, true}) {
            ref.norm(z, gamma, T, silu);
            Dev<float> d_s(q, st_in);
            pf::gdn_recurrence(d_s.p, d_h.p, d_gate.p, d_beta.p, d_z.p, d_gamma.p, EPS, d_y.p, d_y16.p, T, stream, hk, hv, silu);
            q.wait_and_throw();
            const std::vector<float> st = d_s.get();
            const bool fp32 = std::memcmp(st.data(), st_fp32.data(), ns * sizeof(float)) == 0;
            const std::vector<uint16_t> y16 = d_y16.get();
            std::vector<float> y(ny);
            for (size_t i = 0; i < ny; ++i) y[i] = strata::kernels::f32_from_f16(y16[i]);
            const Err es = rel_err(st, ref.state), ey = rel_err(y, ref.y);
            std::snprintf(detail, sizeof detail, "state %.3g, normalized output %.3g", es.worst, ey.worst);
            char name[64];
            std::snprintf(name, sizeof name, "with the output (%s, %s gate)", fp32 ? "FP32 kernels" : "chunked", silu ? "SiLU" : "sigmoid");
            tally.line(name, detail,
                       fp32 ? es.within(kTolFp32) && ey.within(kTolF16Out) : es.within(kTolChunked) && ey.within(kTolChunked));
        }
        // the scan
        if (T >= kScanMin) {
            Dev<float> d_s(q, st_in);
            const bool ran = pf::gdn_rec_tc(q, hk, hv, d_s.p, d_h.p, d_gate.p, d_beta.p, d_y.p, T);
            q.wait_and_throw();
            if (ran) {
                const Err es = rel_err(d_s.get(), ref.state), eo = rel_err(d_y.get(), ref.o);
                std::snprintf(detail, sizeof detail, "state %.3g, output %.3g", es.worst, eo.worst);
                tally.line("the scan (STRATA_GDN_TC)", detail, es.within(kTolFp32) && eo.within(kTolFp32));
            } else {
                ++tally.skipped;
                std::printf("    %-34s no scan for this shape on this GPU (skipped)\n", "the scan (STRATA_GDN_TC)");
            }
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    int64_t T = 2100;
    const char *chunked = nullptr, *keyhead = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--tokens") && i + 1 < argc) T = std::strtoll(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "--chunked") && i + 1 < argc) chunked = argv[++i];
        else if (!std::strcmp(argv[i], "--keyhead") && i + 1 < argc) keyhead = argv[++i];
        else { std::fprintf(stderr, "usage: gdn_prompt_parity [--tokens T] [--chunked 0] [--keyhead 0]\n"); return 2; }
    }
    if (T < 1) { std::fprintf(stderr, "gdn_prompt_parity: --tokens must be 1 or more\n"); return 2; }
    // the switches are read once, at the first recurrence: this run's own, not the caller's environment
    set_env("STRATA_GDN_CHUNKED", chunked);
    set_env("STRATA_GDN_KEYHEAD", keyhead);
    set_env("STRATA_GDN_TC", nullptr);   // the scan is called by name below, the other runs keep their own kernels
    set_env("STRATA_GDN_REC_HEADS", nullptr);

    void* stream = strata::gpu::stream_create();
    sycl::queue& q = strata::core::Runtime::get().stream(stream);
    const int chk = (int) strata::core::kCompiledGeometry.ssm_k_heads, chv = (int) strata::core::kCompiledGeometry.ssm_v_heads;
    std::printf("gdn_prompt_parity (compiled shape %s; STRATA_GDN_CHUNKED %s, STRATA_GDN_KEYHEAD %s)\n",
                strata::core::kGeometryName, chunked != nullptr ? chunked : "unset", keyhead != nullptr ? keyhead : "unset");
    std::vector<std::pair<int, int>> shapes = {{chk, chv}};
    if (chk != strata::prefill::kGdnPlainKeyHeads || chv != strata::prefill::kGdnPlainValueHeads)
        shapes.push_back({strata::prefill::kGdnPlainKeyHeads, strata::prefill::kGdnPlainValueHeads});
    Tally tally;
    for (const auto& [hk, hv] : shapes) check(q, stream, hk, hv, {T, 45}, tally);
    std::printf(tally.fails ? "gdn_prompt_parity: %d of %d FAILED" : "gdn_prompt_parity: %d of %d failed", tally.fails, tally.runs);
    std::printf(tally.skipped ? ", %d skipped\n" : "\n", tally.skipped);
    return tally.fails ? 1 : 0;
}
