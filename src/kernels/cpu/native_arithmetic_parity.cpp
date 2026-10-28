// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// A token's native expert result must not depend on how many other tokens share its group.
// Compare the invariant entry points with the production singleton dispatch, including SwiGLU.
// Synthetic packed blocks only: no GPU, model download or private fixtures.
#include "ggml.h"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/tq_cpu.hpp"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include <utility>

namespace cpu = strata::kernels::cpu;
// NVFP4 (ggml type 40) is not among XeStrata's CPU formats: the tail branches below never run
constexpr int kNvfp4Type = 40;
namespace {
constexpr int width = 8, hidden = 512;
size_t checked = 0;
bool equal(const std::vector<float> &actual, const std::vector<float> &expected) {
    for (size_t i = 0; i < actual.size(); ++i) {
        ++checked;
        if (!std::isfinite(actual[i]) || std::memcmp(&actual[i], &expected[i], sizeof(float)))
            return false;
    }
    return true;
}

bool check_format(int gu, int down, int ff, bool folded = false) {
    cpu::NativeFmt fmt;
    std::string error;
    if (!cpu::native_fmt(gu, down, hidden, ff, fmt, error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return false;
    }
    std::vector<float> gu_signs(hidden), down_signs(ff);
    if (folded) {
        for (int i = 0; i < hidden; ++i) gu_signs[i] = i % 3 ? 1.f : -1.f;
        for (int i = 0; i < ff; ++i) down_signs[i] = i % 5 ? 1.f : -1.f;
        fmt.had_block = 128;
        fmt.had_gu = fmt.had_d = true;
        fmt.gu_signs = gu_signs.data();
        fmt.d_signs = down_signs.data();
    }
    if (std::getenv("STRATA_ARITH_CASE"))
        std::fprintf(stderr, "fmt: bytes=%zu act_bytes=%zu h_bytes=%zu gu_row=%zu up_off=%zu d_row=%zu down_off=%zu gu_act=%d d_act=%d n_embd=%lld n_ff=%lld\n",
                     (size_t) fmt.bytes, (size_t) fmt.act_bytes, (size_t) fmt.h_bytes, (size_t) fmt.gu_row, (size_t) fmt.up_off,
                     (size_t) fmt.d_row, (size_t) fmt.down_off, fmt.gu_act, fmt.d_act, (long long) fmt.n_embd, (long long) fmt.n_ff);
    std::mt19937 rng(20261016 + gu * 100 + down + ff);
    std::vector<uint8_t> blob(fmt.bytes);
    for (auto &byte : blob)
        byte = static_cast<uint8_t>(rng());
    // Use finite block scales and random packed values. NVFP4 has four UE4M3 sub-block scales.
    const uint16_t scale = 0x2400; // 2^-6; exclude NaN/Inf scales from the random bytes.
    auto scales = [&](size_t first, size_t last, int type) {
        const size_t block = cpu::tq_supported(type) ? cpu::tq_block_bytes(type) : ggml_type_size(static_cast<ggml_type>(type));
        for (size_t at = first; at < last; at += block) {
            if (type == kNvfp4Type) std::memset(blob.data() + at, 0x28, 4); // finite UE4M3 sub-block scales
            else {
                std::memcpy(blob.data() + at, &scale, sizeof(scale));
                if (type == 12 || type == 7) std::memcpy(blob.data() + at + 2, &scale, sizeof(scale));
            }
        }
    };
    scales(0, fmt.down_off, gu);
    scales(fmt.down_off, fmt.tail_off ? fmt.tail_off : blob.size(), down);
    if (fmt.tail_off) {
        const float tail[4] = {gu == kNvfp4Type ? .5f : 1.f, gu == kNvfp4Type ? .75f : 1.f,
                               down == kNvfp4Type ? 1.25f : 1.f, 0.f};
        std::memcpy(blob.data() + fmt.tail_off, tail, sizeof(tail));
    }

    // the activations 64-byte aligned, as the pool's buffers are (the canonical rows read them with aligned loads)
    std::vector<std::vector<uint8_t>> act(width, std::vector<uint8_t>(fmt.act_bytes + 64));
    std::vector<std::vector<uint8_t>> hq(width, std::vector<uint8_t>(fmt.h_bytes + 64));
    auto aligned = [](std::vector<uint8_t>& v) {
        return reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(v.data()) + 63) & ~uintptr_t{63});
    };
    std::vector<std::vector<float>> gates(width, std::vector<float>(ff)), downs(width, std::vector<float>(hidden));
    auto solo_gates = gates, solo_downs = downs;
    const void *ap[width], *hp[width];
    float *gp[width], *dp[width], *sg[width], *sd[width];
    for (int t = 0; t < width; ++t) {
        ap[t] = aligned(act[t]);
        hp[t] = aligned(hq[t]);
        gp[t] = gates[t].data();
        dp[t] = downs[t].data();
        sg[t] = solo_gates[t].data();
        sd[t] = solo_downs[t].data();
    }
    for (float amplitude : {0.f, .01f, .3f, 3.f}) {
        std::vector<float> x(hidden), h(ff);
        for (int t = 0; t < width; ++t) {
            for (auto &v : x)
                v = (static_cast<int>(rng() % 2049) - 1024) * amplitude / 1024.f;
            for (auto &v : h)
                v = (static_cast<int>(rng() % 2049) - 1024) * amplitude / 1024.f;
            cpu::native_quant_act(fmt, x.data(), aligned(act[t]));
            cpu::native_quant_h(fmt, h.data(), aligned(hq[t]));
            cpu::native_gu_rows(fmt, blob.data(), ap + t, 1, sg + t, 0, ff);
            cpu::native_down_rows(fmt, blob.data(), hp + t, 1, sd + t, 0, hidden);
        }
        for (int nt = 1; nt <= width; ++nt) {
            cpu::native_gu_rows_invariant(fmt, blob.data(), ap, nt, gp, 0, ff);
            cpu::native_down_rows_invariant(fmt, blob.data(), hp, nt, dp, 0, hidden);
            for (int t = 0; t < nt; ++t) {
                const bool g_ok = equal(gates[t], solo_gates[t]), d_ok = equal(downs[t], solo_downs[t]);
                if (!g_ok || !d_ok) {
                    const auto& a = g_ok ? downs[t] : gates[t];
                    const auto& b = g_ok ? solo_downs[t] : solo_gates[t];
                    size_t i = 0;
                    while (i < a.size() && std::isfinite(a[i]) && std::memcmp(&a[i], &b[i], sizeof(float)) == 0) ++i;
                    std::fprintf(stderr, "native arithmetic mismatch: gu=%d down=%d ff=%d nt=%d token=%d scale=%g: %s row %zu: %g vs solo %g\n",
                                 gu, down, ff, nt, t, amplitude, g_ok ? "down" : "gate/up", i, i < a.size() ? a[i] : 0.f, i < b.size() ? b[i] : 0.f);
                    return false;
                }
            }
        }
    }
    return true;
}
} // namespace


int main() {
    // STRATA_ARITH_CASE=gu,down,ff,folded: one case only (narrowing a failure down)
    if (const char* one = std::getenv("STRATA_ARITH_CASE")) {
        int gu = 0, down = 0, ff = 0, folded = 0;
        if (std::sscanf(one, "%d,%d,%d,%d", &gu, &down, &ff, &folded) != 4) return 2;
        return check_format(gu, down, ff, folded != 0) ? 0 : 1;
    }
    // 16/17 and IQ4_XS exercise the per-token legacy fallback as well as optimized 18/21/22.
    for (int gu : {16, 17, 18, 21, 22, 23}) {
        if (!check_format(gu, 20, 64) || !check_format(gu, 20, 96) || !check_format(gu, 42, 64))
            return 1;
    }
    // (NVFP4, ggml type 40, and its scale tails: not in XeStrata's CPU kernels)
    // STRATA_KQ256 selects grouped Q4_K/Q5_1/Q8_0 kernels only when their target arithmetic matches.
    if (!check_format(12, 7, 64) || !check_format(12, 8, 64)) return 1;
    // 0.2 adds Trellis formats and folded expert inputs. Exercise both their
    // production singleton dispatch and every grouped width through the new API.
    for (int gu : {144, 145, 146})
        for (int down : {20, 144, 145, 146})
            for (bool folded : {false, true})
                if (!check_format(gu, down, 256, folded)) return 1;
    if (!check_format(22, 20, 256, true)) return 1;
    std::printf("native_arithmetic_parity: %zu finite bitwise comparisons passed\n", checked);
}
