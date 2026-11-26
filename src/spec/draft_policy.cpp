// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/spec/draft_policy.cpp - see include/strata/spec/draft_policy.hpp.
#include "strata/spec/draft_policy.hpp"
#include "strata/platform/env.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <vector>

namespace strata::spec {
namespace {

// The shape of a round's cost by window size, relative to one token, used only for sizes not measured yet (the
// measured round times replace it). Between the RTX 5070's measured curves: +10 ms per token with every missed expert
// on the CPU (bench/results/2026-09-27-spec/window-cost), flatter with the default CPU/DMA split.
// Past 8 tokens (a wide window; never measured there) the last step goes on.
constexpr double kShape[DraftPolicy::kMaxT + 1] = {0.0, 1.0, 1.35, 1.7, 2.05, 2.45, 2.85, 3.25, 3.6,
                                                   3.95, 4.3, 4.65, 5.0, 5.35, 5.7, 6.05, 6.4};
constexpr double kCostAlpha = 0.1;    // EMA weight of a new round time
constexpr double kTokAlpha = 0.05;    // EMA weight of a new MTP window outcome
constexpr double kCalDecay = 0.995;   // the MTP probability -> acceptance table fades slowly
constexpr double kDecay = 0.97;       // lookup counts: older windows fade
// Before a bucket has data: the longer the match, the likelier its continuation (llama.cpp's lookup decoding gates
// on the same thing); worth 4 observations, so a few real windows override it
constexpr double kPriorQ[DraftPolicy::kBuckets] = {0.75, 0.88, 0.93, 0.96};
constexpr double kPriorN = 4.0;
constexpr int kProbes = 3;       // a lookup window size is tried this often before its guessed cost can veto it
// A size not measured for this many rounds is tried again when a confident lookup could use it: its cost was set in
// one stretch (a prompt's cache refill, say) and, priced high, it would never be measured again (#1252)
constexpr double kStaleRounds = 300.0;
// The MTP's window: a draft's acceptance before any is observed (the same 0.7 as prior_tokens), and the chance the
// drafts must have of holding for a size whose cost is only guessed to be tried (as chain()'s 0.6 for a lookup)
constexpr double kDraftPrior = 0.7;
constexpr double kProbeHeld = 0.5;
// STRATA_SPEC_COST (set_robust), XeStrata-890M.  A round's time has a floor, the work of its rows, and what disturbs a
// round (another process, a clock step, a capture in the draft that follows) only adds to it.  So a size's first
// kProbes times count by their least, and after them a time below the mean moves it faster (kCostAlphaDown) than one
// above (kCostAlpha), which is also cut at kHiccup times the mean.  With the running mean of before, one disturbed
// round among a size's three probes left it priced a third of the disturbance too high, and a size priced out is
// not measured again.  The 890M's greedy 6 chats (--spec 4) showed starts whose windows stayed too narrow or too
// wide to the end of the process: 34.0-35.6 tok/s against 36.4-38.1, 4 of about 60 starts, and none in the
// 14 starts with this (2026-11-16; which round set the wrong time was not seen).  The MTP chain
// also tries a size again whose time is kStaleChain rounds old, when its drafts are likely to hold (kProbeHeld):
// a time that is wrong for another reason is then corrected within a few hundred rounds.
constexpr double kCostAlphaDown = 0.25;
constexpr double kHiccup = 1.25;
constexpr double kStaleChain = 128.0;
// The rate rule (STRATA_SPEC_COST, XeStrata-890M): the MTP chain's width is the one with the most expected tokens less
// the long-run rate times its round time, E - lambda * cost, and not the one with the best E / cost of its own.  Over
// a stream of windows that differ, tokens per ms is the largest when every window maximizes E - lambda* cost with
// lambda* the stream's own rate (a ratio of sums: Dinkelbach's transform): a window of unsure drafts is then kept
// narrower than its own ratio asks (a row has to bring lambda times its ms, and the ratio of a poor window is below
// lambda), a window of sure drafts wider.  lambda is the tokens committed over the round times of the last few
// hundred timed rounds; until kRateMin rounds were timed the window's own ratio decides, as before.
// STRATA_SPEC_RATE=0: the ratio always.
constexpr double kRateDecay = 0.995;
constexpr double kRateMin = 16.0;

// Price a table whose sizes were not all reached by real rounds.  A wider window cannot cost less than a narrower
// one: a size without real rounds is priced on the straight line between the nearest real sizes around it, and the
// sizes past either end take that end's price.  A guess made from one stretch of live rounds used to be able to price
// size 3 above the measured size 4 (B70 and RTX 4070, 2026-11-24), which made the fixed table choose narrower windows
// than live costs did.  With no real size at all, keep the live guesses but still stop them from falling.
void monotone_prices(double* v, int n, unsigned real) {
    if (n <= 0) return;
    int first = 0, last = 0;
    for (int t = 1; t <= n; ++t)
        if ((real >> (t - 1) & 1u) && v[t] > 0.0) {
            if (first == 0) first = t;
            last = t;
        }
    if (first == 0) {
        for (int t = 1; t <= n; ++t) {
            if (!(v[t] > 0.0)) v[t] = t > 1 ? v[t - 1] : 1.0;
            if (t > 1) v[t] = std::max(v[t], v[t - 1]);
        }
        return;
    }
    for (int t = 1; t < first; ++t) v[t] = v[first];
    for (int t = last + 1; t <= n; ++t) v[t] = v[last];
    int prev = first;
    for (int t = first + 1; t <= last; ++t) {
        if ((real >> (t - 1) & 1u) && v[t] > 0.0) { prev = t; continue; }
        int next = t;
        while (next <= last && !((real >> (next - 1) & 1u) && v[next] > 0.0)) ++next;
        v[t] = v[prev] + (v[next] - v[prev]) * double(t - prev) / double(next - prev);
    }
    for (int t = 2; t <= n; ++t) v[t] = std::max(v[t], v[t - 1]);   // two real sizes can also be inverted
}

// XESTRATA_SPEC_FIXED=1 and XESTRATA_SPEC_FIXED_TABLE="T1,T2,.." (round ms of windows of 1, 2, .. rows): an expert's
// fixed table for every context bucket, read once per process (the table of --draft-costs is the normal source)
struct FixedEnv {
    bool on = false;
    double ms[DraftPolicy::kMaxT] = {};
    int n = 0;
};
const FixedEnv& fixed_env() {
    static const FixedEnv cfg = [] {
        FixedEnv c;
        const char* e = strata::env_get("SPEC_FIXED");
        c.on = e != nullptr && e[0] == '1';
        if (const char* tab = strata::env_get("SPEC_FIXED_TABLE"); c.on && tab != nullptr && tab[0] != '\0') {
            for (const char* q = tab; *q != '\0' && c.n < DraftPolicy::kMaxT;) {
                char* e = nullptr;
                c.ms[c.n++] = std::strtod(q, &e);
                q = e != q ? e : q + 1;
                while (*q != '\0' && *q != ',') ++q;
                if (*q == ',') ++q;
            }
        }
        return c;
    }();
    return cfg;
}

}  // namespace

// ---------------------------------------------------------------------------------------------------- the table file

static_assert(CostFile::kMaxT == DraftPolicy::kMaxT && CostFile::kCtxBuckets == DraftPolicy::kCtxBuckets, "one table");

namespace {
constexpr const char* kCostFileHead = "xestrata-draft-costs 1";
// the path as UTF-8 (the server passes it so); on Windows a plain std::string would be read in the ANSI code page
std::filesystem::path utf8_path(const std::string& s) { return std::filesystem::path(std::u8string(s.begin(), s.end())); }
std::string one_line(std::string s) {
    for (char& c : s)
        if (c == '\n' || c == '\r') c = ' ';
    return s;
}
}  // namespace

// The numbers of a line after its first `from` characters, spaces or commas between them; false on anything else
bool numbers(const std::string& line, size_t from, std::vector<double>& v) {
    v.clear();
    const char* q = line.c_str() + from;
    for (;;) {
        while (*q == ' ' || *q == ',') ++q;
        if (*q == '\0') return true;
        char* e = nullptr;
        const double x = std::strtod(q, &e);
        if (e == q || !std::isfinite(x)) return false;
        v.push_back(x);
        q = e;
    }
}
bool whole(double x, double lo, double hi) { return x >= lo && x <= hi && x == std::floor(x); }

bool read_cost_file(const std::string& path, const std::string& key, CostFile& out, std::string& why) {
    out = CostFile{};
    std::ifstream f(utf8_path(path), std::ios::binary);
    if (!f) { why = "no table file yet"; return false; }
    std::string line;
    if (!std::getline(f, line) || line != kCostFileHead) { why = "not a table file of this format"; return false; }
    if (!std::getline(f, line) || line.rfind("key ", 0) != 0) { why = "the file has no key"; return false; }
    if (line.substr(4) != one_line(key)) { why = "built on another machine or with other settings"; return false; }
    constexpr int B = CostFile::kCtxBuckets, T = CostFile::kMaxT;
    bool ended = false, any = false;
    std::vector<double> v;
    auto starts = [&](const char* word) { return line.rfind(word, 0) == 0; };
    while (std::getline(f, line)) {
        if (line == "end") { ended = true; break; }
        if (line == "fresh 1") { out.fresh = true; continue; }
        if (starts("reqs ")) {
            if (!numbers(line, 5, v) || v.size() != 1 || !whole(v[0], 0, 1e6)) { why = "a malformed request count"; return false; }
            out.reqs = (int) v[0];
        } else if (starts("real ")) {
            if (!numbers(line, 5, v) || v.size() != 2 || !whole(v[0], 0, B - 1) || !whole(v[1], 0, 4294967295.0)) { why = "a malformed real mark"; return false; }
            out.real[(size_t) v[0]] = (unsigned) v[1];
        } else if (starts("rc ")) {
            if (!numbers(line, 3, v) || v.size() != 3 || !whole(v[0], 0, B - 1) || !whole(v[1], 1, T) || !(v[2] > 0.0) || !(v[2] < 1e5)) {
                why = "a malformed real cost";
                return false;
            }
            out.real_cost[(size_t) v[0]][(size_t) v[1]] = v[2];
        } else if (starts("cal ")) {
            if (!numbers(line, 4, v) || v.size() < 4 || !whole(v[0], 0, B - 1) || !whole(v[1], 1, T) || !whole(v[2], 0, CalAcc::kVals)
                || v.size() != 4 + (size_t) v[2] || !(v[3] >= 0.0) || !(v[3] < 1e9)) {
                why = "a malformed calibration line";
                return false;
            }
            CalAcc& ca = out.cal[(size_t) v[0]][(size_t) v[1]];
            ca.n = ca.nv = (int) v[2];
            ca.sum = v[3];
            for (int k = 0; k < ca.nv; ++k) {
                if (!(v[4 + (size_t) k] > 0.0) || !(v[4 + (size_t) k] < 1e6)) { why = "a malformed calibration line"; return false; }
                ca.vals[(size_t) k] = v[4 + (size_t) k];
            }
        } else if (starts("bucket ")) {
            if (!numbers(line, 7, v) || v.size() < 2 || v.size() > (size_t) T + 1 || !whole(v[0], 0, B - 1)) { why = "a malformed table"; return false; }
            const size_t b = (size_t) v[0];
            for (size_t t = 1; t < v.size(); ++t) {
                if (!(v[t] > 0.0) || !(v[t] < 1e5)) { why = "a malformed table"; return false; }
                out.table[b][t] = v[t];
            }
            out.n[b] = (int) v.size() - 1;
            any = true;
        } else {
            why = "a malformed line";
            return false;
        }
    }
    if (!ended || any == out.fresh) { why = "the file is cut short"; return false; }
    out.key = key;
    return true;
}

bool write_cost_file(const std::string& path, const CostFile& c, std::string& why) {
    namespace fs = std::filesystem;
    std::error_code ec;
    const fs::path p = utf8_path(path);
    if (p.has_parent_path()) fs::create_directories(p.parent_path(), ec);
    if (ec) { why = "could not make the folder (" + ec.message() + ")"; return false; }
    fs::path tmp = p;
    tmp += ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f) { why = "could not write the file"; return false; }
        f << kCostFileHead << "\nkey " << one_line(c.key) << "\n";
        char b[32];
        for (int k = 0; k < CostFile::kCtxBuckets; ++k) {
            if (c.n[(size_t) k] <= 0) continue;
            f << "bucket " << k << " ";
            for (int t = 1; t <= c.n[(size_t) k]; ++t) {
                std::snprintf(b, sizeof b, "%s%.3f", t > 1 ? "," : "", c.table[(size_t) k][(size_t) t]);
                f << b;
            }
            f << "\n";
        }
        if (c.reqs > 0) f << "reqs " << c.reqs << "\n";
        if (c.fresh) f << "fresh 1\n";
        for (int k = 0; k < CostFile::kCtxBuckets; ++k) {
            if (c.real[(size_t) k] != 0u) f << "real " << k << " " << c.real[(size_t) k] << "\n";
            for (int t = 1; t <= CostFile::kMaxT; ++t) {
                const bool real = c.real[(size_t) k] >> (t - 1) & 1u;
                if (real && c.real_cost[(size_t) k][(size_t) t] > 0.0) {
                    std::snprintf(b, sizeof b, "%.4f", c.real_cost[(size_t) k][(size_t) t]);
                    f << "rc " << k << " " << t << " " << b << "\n";
                }
                const CalAcc& a = c.cal[(size_t) k][(size_t) t];
                if (a.n <= 0 || real) continue;
                std::snprintf(b, sizeof b, "%.4f", a.sum);
                f << "cal " << k << " " << t << " " << a.nv << " " << b;
                for (int i = 0; i < a.nv; ++i) {
                    std::snprintf(b, sizeof b, " %.4f", a.vals[(size_t) i]);
                    f << b;
                }
                f << "\n";
            }
        }
        f << "end\n";
        f.flush();
        if (!f) { why = "could not write the file"; fs::remove(tmp, ec); return false; }
    }
    fs::rename(tmp, p, ec);   // replaces an older file whole (MoveFileEx with REPLACE_EXISTING on Windows)
    if (ec) { why = "could not replace the file (" + ec.message() + ")"; fs::remove(tmp, ec); return false; }
    return true;
}

std::string draft_costs_key(int argc, char** argv, const std::string& devices, const std::string& extra) {
    std::string k = devices;
    const char* sc = strata::env_get("SPEC_COST");
    k += std::string("|spec_cost ") + (sc != nullptr ? sc : "auto");
    k += "|args";
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--draft-costs" || a == "--draft-costs-key") { ++i; continue; }
        std::error_code ec;
        const std::filesystem::path p = utf8_path(a);
        const size_t cut = a.find_last_of("/\\");
        std::string base = cut == std::string::npos ? a : a.substr(cut + 1);
        if (base.empty()) base = a;   // a path ending in a separator: kept whole
        if (a.size() > 1 && a[0] != '-' && std::filesystem::is_regular_file(p, ec))
            k += " " + base + ":" + std::to_string((unsigned long long) std::filesystem::file_size(p, ec));
        else if (a.size() > 1 && a[0] != '-' && std::filesystem::is_directory(p, ec))
            k += " " + base + "/";
        else
            k += " " + a;
    }
    if (!extra.empty()) k += "|server " + extra;
    return one_line(k);
}

// ---------------------------------------------------------------------------------------------------- DraftPolicy

DraftPolicy::DraftPolicy(int max_t, double margin, bool source_costs)
    : max_t_(std::clamp(max_t, 1, kMaxT)), margin_(margin), source_costs_(source_costs) {
    // XESTRATA_SPEC_COST=1: the cost-aware rule (set_robust) for a caller that does not set it; opt-in (upstream's
    // auto, from 6K tokens of context on, is not taken: XeStrata-890M keeps it opt-in, on by default on the HIP
    // integrated GPU through spec_cost_default())
    const char* v = strata::env_get("SPEC_COST");
    robust_ = v != nullptr && v[0] != '\0' && std::strtol(v, nullptr, 10) != 0;
    const FixedEnv& f = fixed_env();
    if (f.on)
        for (int k = 0; k < kCtxBuckets; ++k) set_fixed_costs(f.n > 0 ? f.ms : nullptr, f.n, k);
}

int DraftPolicy::bucket(int match) {
    return match < 6 ? 0 : match < 12 ? 1 : match < 24 ? 2 : 3;
}

double DraftPolicy::lookup_rate(int match) const {
    const int b = bucket(match);
    return (ok_[b] + kPriorN * kPriorQ[b]) / (ok_[b] + bad_[b] + kPriorN);
}

double DraftPolicy::cost_ms(int t, bool lookup) const {
    t = std::clamp(t, 1, kMaxT);
    if (source_costs_ && lookup && !fixed_ && lookup_cost_n_[bk_][t] > 0) return lookup_cost_[bk_][t];   // a fixed table prices both
    if (cost_n_[bk_][t] > 0) return cost_[bk_][t];
    // scale from the nearest measured size below: a round's fixed part flattens the measured curve at its small end,
    // and scaled from there the wider sizes came out too dear to be tried (Gyro-M on the RX 9060 XT: 46 ms at one
    // token, 57 at four, the six guessed at 98 against its 74)
    for (int u = t - 1; u >= 1 && !scale_all_; --u)
        if (cost_n_[bk_][u] > 0) return cost_[bk_][u] * kShape[t] / kShape[u];
    // none below: scale from the measured sizes, weighting each by how often it was seen
    double num = 0.0, den = 0.0;
    for (int u = 1; u <= kMaxT; ++u)
        if (cost_n_[bk_][u] > 0) {
            const double w = std::min(cost_n_[bk_][u], 20.0);
            num += w * cost_[bk_][u] * kShape[t] / kShape[u];
            den += w;
        }
    return den > 0 ? num / den : kShape[t];
}

double DraftPolicy::clamp_sample(int t, double ms, bool lookup) const {
    if (!robust_) return ms;
    // a size's first times are taken as they are (update_cost keeps their least; the caller leaves out the round that
    // captured the size's graph); later ones are cut at kHiccup times what is known
    if ((source_costs_ && lookup ? lookup_cost_n_ : cost_n_)[bk_][t] < kProbes) return ms;
    return std::min(ms, kHiccup * cost_ms(t, lookup));
}

void DraftPolicy::update_cost(double& mean, double& n, double ms) const {
    if (!robust_) {
        mean = n > 0 ? (1.0 - kCostAlpha) * mean + kCostAlpha * ms : ms;
    } else if (n < 1.0) {
        mean = ms;   // the first time, or the first after another context's value stood in (set_context)
        n = 0.0;
    } else if (n < kProbes) {
        mean = std::min(mean, ms);
    } else {
        const double alpha = ms < mean ? kCostAlphaDown : kCostAlpha;
        mean = (1.0 - alpha) * mean + alpha * ms;
    }
    n += 1.0;
}

double DraftPolicy::add_cost(int t, double round_ms, bool lookup) {
    const bool apart = source_costs_ && lookup;
    if (calib_on_ && !apart) calib_add(bk_, t, round_ms);   // the table is the MTP windows' (cost_)
    round_ms = clamp_sample(t, round_ms, lookup);
    if (fixed_) return round_ms;
    // source_costs: a lookup window's time goes into the lookup table, so neither source prices the other's rounds
    update_cost((apart ? lookup_cost_ : cost_)[bk_][t], (apart ? lookup_cost_n_ : cost_n_)[bk_][t], round_ms);
    last_[bk_][t] = rounds_;
    return round_ms;
}

void DraftPolicy::seed_costs(const double* ms, int n, double weight, int bucket) {
    bucket = std::clamp(bucket, 0, kCtxBuckets - 1);
    for (int t = 1; t <= std::min(n, kMaxT); ++t)
        if (ms[t - 1] > 0.0 && cost_n_[bucket][t] == 0) {
            cost_[bucket][t] = ms[t - 1];
            cost_n_[bucket][t] = weight;
        }
}

void DraftPolicy::set_context(long long pos) {
    if (!robust_) return;
    const int nb = pos < 6000 ? 0 : pos < 24000 ? 1 : 2;
    if (nb == bk_) return;
    fill_bucket(nb);
    bk_ = nb;
}

void DraftPolicy::fill_bucket(int nb) {
    bool has = false;
    for (int t = 1; t <= kMaxT; ++t) has = has || cost_n_[nb][t] > 0;
    if (!has) {   // nothing measured at this length yet: the nearest bucket that has data is the prior (one sample's worth)
        for (int d = 1; d < kCtxBuckets && !has; ++d)
            for (const int src : {nb - d, nb + d}) {
                if (src < 0 || src >= kCtxBuckets || has) continue;
                for (int t = 1; t <= kMaxT; ++t)
                    if (cost_n_[src][t] > 0) {
                        cost_[nb][t] = cost_[src][t];
                        // a stand-in: this length's first time replaces it (update_cost); a fixed table's copy is fixed too
                        cost_n_[nb][t] = fixed_ ? cost_n_[src][t] : 0.5;
                        last_[nb][t] = fixed_ ? last_[src][t] : rounds_;
                        has = true;
                    }
            }
    }
}

std::string DraftPolicy::cost_table_text() const {
    std::string out;
    char b[64];
    for (int k = 0; k < kCtxBuckets; ++k) {
        bool any = false;
        for (int t = 1; t <= kMaxT; ++t) any = any || cost_n_[k][t] > 0;
        if (!any) continue;
        out += (out.empty() ? "" : " | ");
        out += k == 0 ? "ctx<6K:" : k == 1 ? "ctx<24K:" : "ctx>=24K:";
        for (int t = 1; t <= kMaxT; ++t)
            if (cost_n_[k][t] > 0) {
                std::snprintf(b, sizeof b, " T%d:%.2f(n=%.0f)", t, cost_[k][t], cost_n_[k][t]);
                out += b;
            }
    }
    return out;
}

bool DraftPolicy::stale_chain(int t) const {
    return robust_ && !fixed_ && t <= kMaxT && cost_n_[bk_][t] > 0 && rounds_ - last_[bk_][t] > kStaleChain;
}

bool DraftPolicy::stale(int t, bool lookup) const {
    return !fixed_ && (source_costs_ && lookup ? lookup_cost_n_ : cost_n_)[bk_][t] > 0 && rounds_ - last_[bk_][t] > kStaleRounds;
}

double DraftPolicy::mtp_tokens(int t) const {
    if (mtp_n_[t] > 0) return mtp_tok_[t];
    return prior_tokens(t);
}

DraftPolicy::Pick DraftPolicy::choose(int t_mtp, int lookup_k, int match) const {
    Pick p;
    p.t = std::clamp(t_mtp, 1, max_t_);
    if (lookup_k <= 0) return p;
    const double base = mtp_tokens(p.t) / cost_ms(p.t);
    const double q = lookup_rate(match);
    double e = 1.0, qi = 1.0, best = 0.0;
    int best_t = 0;
    for (int k = 1; k <= std::min(lookup_k, max_t_ - 1); ++k) {
        qi *= q;
        e += qi;
        const double r = e / cost_ms(k + 1, true);
        if (r > best) { best = r; best_t = k + 1; }
    }
    const int t_full = std::min(lookup_k, max_t_ - 1) + 1;
    if (best_t > 0 && best > base * (1.0 + margin_)) {
        p.lookup = true;
        p.t = best_t;
        // a wider size whose cost has gone stale is re-measured, so one slow stretch cannot price it out for good
        if (best_t < t_full && q >= 0.85 && stale(t_full, true)) p.t = t_full;
        return p;
    }
    // a guessed cost can keep the policy from ever measuring a size: the first few times a confident lookup would
    // need a size not measured yet, it is tried (verification keeps the output; only the one round's speed is at stake)
    const double measured = source_costs_ ? lookup_cost_n_[bk_][t_full] : cost_n_[bk_][t_full] - delayed_n_[bk_][t_full];
    if (!fixed_ && t_full > p.t && (measured + tried_[bk_][t_full] < kProbes || stale(t_full, true)) && q >= 0.85) {
        p.lookup = true;
        p.t = t_full;
    }
    return p;
}

void DraftPolicy::observe(bool lookup, int t, int accepted, int match, double round_ms) {
    t = std::clamp(t, 1, kMaxT);
    if (round_ms > 0) observe_rate(accepted + 1.0, add_cost(t, round_ms, lookup));
    rounds_ += 1.0;
    if (lookup) {
        const int b = bucket(match);
        ok_[b] = kDecay * ok_[b] + accepted;
        bad_[b] = kDecay * bad_[b] + (accepted < t - 1 ? 1.0 : 0.0);
    } else {
        const double got = accepted + 1.0;
        mtp_tok_[t] = mtp_n_[t] > 0 ? (1.0 - kTokAlpha) * mtp_tok_[t] + kTokAlpha * got : got;
        mtp_n_[t] += 1.0;
    }
}

void DraftPolicy::observe_committed(bool lookup, int t, int accepted, int match) {
    t = std::clamp(t, 1, kMaxT);
    rounds_ += 1.0;
    if (lookup) {
        tried_[bk_][t] += 1.0;
        last_[bk_][t] = rounds_;   // tried now: not stale within this generation
        const int b = bucket(match);
        ok_[b] = kDecay * ok_[b] + accepted;
        bad_[b] = kDecay * bad_[b] + (accepted < t - 1 ? 1.0 : 0.0);
    } else {
        const double got = accepted + 1.0;
        mtp_tok_[t] = mtp_n_[t] > 0 ? (1.0 - kTokAlpha) * mtp_tok_[t] + kTokAlpha * got : got;
        mtp_n_[t] += 1.0;
    }
}

void DraftPolicy::defer_cost(int64_t end, int t, double round_ms) {
    if (round_ms > 0) held_.push_back({end, bk_, std::clamp(t, 1, kMaxT), round_ms});
}

void DraftPolicy::apply_costs(int64_t committed) {
    size_t n = 0;
    const int bucket_now = bk_;
    for (; n < held_.size() && held_[n].end < committed; ++n) {
        const HeldCost& h = held_[n];
        bk_ = h.bucket;   // observe()'s update, in the bucket the round ran in
        add_cost(h.t, h.ms);
        if (!fixed_) delayed_n_[bk_][h.t] += 1.0;   // a round already counted by observe_committed
    }
    bk_ = bucket_now;
    held_.erase(held_.begin(), held_.begin() + (std::ptrdiff_t) n);
}

namespace {
int p_bucket(double p) {
    const int b = (int) (p * 8.0);
    return b < 0 ? 0 : b > 7 ? 7 : b;
}
}  // namespace

// Until a bucket has data a draft is as likely to hold as the MTP says (worth kPriorN observations); measured
// acceptance of the MTP chain is not the same as its own softmax (code and thinking text are over-confident).
double DraftPolicy::draft_rate(double p) const {
    const int b = p_bucket(p);
    const double centre = std::clamp(p, 0.0, 1.0);
    return (pa_ok_[b] + kPriorN * centre) / (pa_n_[b] + kPriorN);
}

void DraftPolicy::observe_probs(const float* p, int n, int accepted) {
    // draft j was reached when the drafts before it held; it held when j < accepted
    for (int j = 0; j < n && j <= accepted; ++j) {
        const int b = p_bucket(p[j]);
        pa_ok_[b] = kCalDecay * pa_ok_[b] + (j < accepted ? 1.0 : 0.0);
        pa_n_[b] = kCalDecay * pa_n_[b] + 1.0;
    }
}

void DraftPolicy::observe_rate(double tokens, double ms) {
    rate_tok_ = kRateDecay * rate_tok_ + tokens;
    rate_ms_ = kRateDecay * rate_ms_ + ms;
    rate_n_ = kRateDecay * rate_n_ + 1.0;
}

bool DraftPolicy::rate_rule() const {
    static const bool on = [] {
        const char* e = std::getenv("STRATA_SPEC_RATE");
        return e == nullptr || std::strtol(e, nullptr, 10) != 0;
    }();
    return on && robust_ && rate_n_ >= kRateMin && rate_ms_ > 0.0;
}

bool DraftPolicy::keep_draft(ChainState& st, int idx, float p, int cap) const {
    cap = std::clamp(cap, 1, max_t_);
    if (idx == 0) {
        st = ChainState{};
        st.best_v = 1.0 / cost_ms(1);
        st.best_c = cost_ms(1);
    }
    const double a = draft_rate(p);
    st.q *= a;
    st.e += st.q;
    const int rows = idx + 2;
    // the rate rule: a wider window is better when its added tokens beat the long-run rate times its added time
    const bool by_rate = rate_rule();
    const double lam = by_rate ? rate_tok_ / rate_ms_ * (1.0 + margin_ / 3.0) : 0.0;
    const auto beats = [&](double e, double c) { return e - st.best_e > lam * (c - st.best_c); };
    if (rows <= cap) {
        const double c = cost_ms(rows), v = st.e / c;
        if (by_rate ? beats(st.e, c) : v > st.best_v * (1.0 + margin_ / 3.0)) {
            st.best_v = v;
            st.best_e = st.e;
            st.best_c = c;
            st.best_rows = rows;
        }
        // a width not measured yet is tried a few times when the chain is likely to hold that far, and one whose time
        // has grown old once in a while, when its drafts are likely to hold (kStaleChain)
        if ((!fixed_ && cost_n_[bk_][rows] < kProbes && st.q >= 0.15) || (stale_chain(rows) && st.q >= kProbeHeld)) st.best_rows = rows;
        if (st.best_rows == rows && st.prefix_rows == rows - 1) st.prefix_rows = rows;
    }
    st.kept = st.best_rows - 1;
    if (rows + 1 > cap) return false;                       // no row left for another draft
    // one or two more drafts, each as likely to hold as this one did, could they beat the best width so far?
    const double e1 = st.e + st.q * a, e2 = e1 + st.q * a * a;
    const double v1 = e1 / cost_ms(rows + 1);
    const double v2 = rows + 2 <= cap ? e2 / cost_ms(rows + 2) : 0.0;
    const bool pays = by_rate ? beats(e1, cost_ms(rows + 1)) || (rows + 2 <= cap && beats(e2, cost_ms(rows + 2)))
                              : std::max(v1, v2) > st.best_v * (1.0 + margin_ / 3.0);
    return pays || (!fixed_ && cost_n_[bk_][rows + 1] < kProbes && st.q * a >= 0.15) ||
           (stale_chain(rows + 1) && st.q * a >= kProbeHeld);
}

// The draft layer's probability is not the model's: the calibration is the accepted drafts over the probabilities
// of the drafts tested (a draft is tested only while those before it held, so each ratio is conditional on them)
double DraftPolicy::draft_q(float p) const {
    const double calib = (dq_ok_ + kPriorN * kDraftPrior) / (dq_p_ + kPriorN * kDraftPrior);
    return std::clamp((double) p * calib, 0.0, 1.0);
}

double DraftPolicy::draft_q_mean() const { return (dq_ok_ + kPriorN * kDraftPrior) / (dq_n_ + kPriorN); }

// A window of t tokens, expecting `expected` committed tokens, whose drafts hold with probability `held`: does one
// more draft of acceptance q raise its tokens per ms?  Yes when the marginal tokens over the marginal ms beat the
// window's rate (the greedy step of a ratio).  A size whose cost is only a guess is measured instead, when likely.
bool DraftPolicy::mtp_step_pays(int t, double held, double q, double expected) const {
    if (t + 1 > max_t_) return false;
    const double gain = held * q;
    if (!fixed_ && (cost_n_[bk_][t + 1] < kProbes || stale(t + 1))) return gain >= kProbeHeld;
    const double dc = cost_ms(t + 1) - cost_ms(t);
    if (dc <= 0.0) return gain > 0.0;   // measured no dearer: the tokens come free
    return gain / dc > expected / cost_ms(t);
}

int DraftPolicy::mtp_window(const float* probs, int max_t) const {
    max_t = std::clamp(max_t, 1, max_t_);
    int t = 1;
    double held = 1.0, expected = 1.0;
    while (t < max_t) {
        const double q = draft_q(probs[t - 1]);
        if (q <= 0.0 || !mtp_step_pays(t, held, q, expected)) break;
        expected += held * q;
        held *= q;
        ++t;
    }
    return t;
}

bool DraftPolicy::mtp_extend(const float* probs, int n, int max_t) const {
    max_t = std::clamp(max_t, 1, max_t_);
    if (n < 0 || n + 1 >= max_t) return false;
    double held = 1.0, expected = 1.0;
    for (int t = 1; t <= n; ++t) {   // the drafts made must pay first, each at its own probability
        const double q = draft_q(probs[t - 1]);
        if (q <= 0.0 || !mtp_step_pays(t, held, q, expected)) return false;
        expected += held * q;
        held *= q;
    }
    return mtp_step_pays(n + 1, held, draft_q_mean(), expected);
}

void DraftPolicy::observe_mtp_drafts(const float* probs, int t, int accepted) {
    dq_ok_ *= kDecay;
    dq_p_ *= kDecay;
    dq_n_ *= kDecay;
    for (int d = 0; d < t - 1 && d <= accepted; ++d) {   // tested: reached while the drafts before it held
        dq_n_ += 1.0;
        dq_p_ += std::clamp((double) probs[d], 0.0, 1.0);
        dq_ok_ += d < accepted ? 1.0 : 0.0;
    }
}

double DraftPolicy::chain_rate(int match) const {
    // prior: a chained draft is a lookup continuing the MTP's drafts - weaker evidence than a plain match of that length
    constexpr double kChainPrior[kBuckets] = {0.45, 0.65, 0.8, 0.9};
    const int b = bucket(match);
    return (cok_[b] + kPriorN * kChainPrior[b]) / (cok_[b] + cbad_[b] + kPriorN);
}

int DraftPolicy::chain(int t_mtp, double p_mtp, int k_avail, int match) const {
    t_mtp = std::clamp(t_mtp, 1, max_t_);
    const int kmax = std::min(k_avail, max_t_ - t_mtp);
    if (kmax <= 0) return 0;
    const double e0 = mtp_tokens(t_mtp), base = e0 / cost_ms(t_mtp), c = chain_rate(match);
    p_mtp = std::clamp(p_mtp, 0.0, 1.0);
    double gain = 0.0, ci = 1.0, best = 0.0;
    int best_k = 0;
    for (int k = 1; k <= kmax; ++k) {
        ci *= c;
        gain += ci;
        const double r = (e0 + p_mtp * gain) / cost_ms(t_mtp + k);
        if (r > best) { best = r; best_k = k; }
    }
    const int t_full = t_mtp + kmax;
    if (best_k > 0 && best > base * (1.0 + margin_)) {
        // As choose(): a profitable shorter chain must not keep a stale full size from being measured again.
        if (best_k < kmax && p_mtp * c >= 0.6 && stale(t_full)) return kmax;
        return best_k;
    }
    // as choose(): a size whose cost is only guessed is tried a few times when the continuation looks likely
    if (!fixed_ && (cost_n_[bk_][t_full] < kProbes || stale(t_full)) && p_mtp * c >= 0.6) return kmax;
    return 0;
}

void DraftPolicy::observe_chain(int t_mtp, int k, int accepted, int match, double round_ms) {
    const int t = std::clamp(t_mtp + k, 1, kMaxT);
    if (round_ms > 0) observe_rate(accepted + 1.0, add_cost(t, round_ms));
    rounds_ += 1.0;
    // the MTP part is an ordinary MTP window as far as its own drafts go
    const int mtp_acc = std::min(accepted, t_mtp - 1);
    const double got = mtp_acc + 1.0;
    mtp_tok_[t_mtp] = mtp_n_[t_mtp] > 0 ? (1.0 - kTokAlpha) * mtp_tok_[t_mtp] + kTokAlpha * got : got;
    mtp_n_[t_mtp] += 1.0;
    if (accepted < t_mtp - 1) return;   // the chain was never reached: it says nothing about the lookup
    const int b = bucket(match), ok = accepted - (t_mtp - 1);
    cok_[b] = kDecay * cok_[b] + ok;
    cbad_[b] = kDecay * cbad_[b] + (ok < k ? 1.0 : 0.0);
}

// ---------------------------------------------------------------------------------------------------- fixed costs

void DraftPolicy::set_fixed_costs(const double* ms, int n, int bucket) {
    if (!fixed_ && !live_saved_) {   // the live costs at this point: start_calibration() may hand them back
        live_cost_ = cost_;
        live_n_ = cost_n_;
        live_saved_ = true;
    }
    fixed_ = true;
    held_table_ = false;
    if (ms == nullptr || n <= 0) return;
    bucket = std::clamp(bucket, 0, kCtxBuckets - 1);
    fixed_n_[(size_t) bucket] = std::min(n, kMaxT);
    for (int t = 1; t <= kMaxT; ++t) {   // the table replaces what was measured; sizes past its end scale from it
        const bool in = t <= n && ms[t - 1] > 0.0;
        cost_[(size_t) bucket][(size_t) t] = in ? ms[t - 1] : 0.0;
        cost_n_[(size_t) bucket][(size_t) t] = in ? 1e6 : 0.0;
        learned_[(size_t) bucket][(size_t) t] = cost_[(size_t) bucket][(size_t) t];
    }
}

void DraftPolicy::request_start() {
    const auto now = std::chrono::steady_clock::now();
    if (!seen_request_ || std::chrono::duration<double>(now - last_end_).count() >= kIdleS) busy_since_ = now;
    seen_request_ = true;
}

void DraftPolicy::request_end() { last_end_ = std::chrono::steady_clock::now(); }

// The price of every size of a table in use: a real size its real rounds' median; a size real requests have not
// reached its learned cost x the mean (real / learned) of the real sizes of 2 or more rows (size 1 stays); then more
// rows never cost less, an inversion repaired from the trusted side (upstream 9448501635)
void DraftPolicy::apply_table() {
    if (!fixed_) return;   // a held table: cost_ is the live cost, not touched
    for (int k = 0; k < kCtxBuckets; ++k) {
        const int n = fixed_n_[(size_t) k];
        if (n <= 0) continue;
        auto& c = cost_[(size_t) k];
        const unsigned real = real_[(size_t) k];
        double sum = 0.0;
        int cnt = 0;
        for (int t = 2; t <= n; ++t)
            if ((real >> (t - 1) & 1u) && learned_[(size_t) k][(size_t) t] > 0.0) {
                sum += real_cost_[(size_t) k][(size_t) t] / learned_[(size_t) k][(size_t) t];
                ++cnt;
            }
        const double r = cnt > 0 ? sum / cnt : 1.0;
        for (int t = 1; t <= n; ++t)
            c[(size_t) t] = (real >> (t - 1) & 1u) ? real_cost_[(size_t) k][(size_t) t] : learned_[(size_t) k][(size_t) t] * (t >= 2 ? r : 1.0);
        for (int t = n - 1; t >= 1; --t) {
            const bool real_t = real >> (t - 1) & 1u, real_next = real >> t & 1u;
            if (c[(size_t) t] > c[(size_t) t + 1] && (!real_t || real_next)) c[(size_t) t] = c[(size_t) t + 1];
        }
        for (int t = 2; t <= n; ++t) c[(size_t) t] = std::max(c[(size_t) t], c[(size_t) t - 1]);
    }
}

void DraftPolicy::start_calibration(const CostFile& cf) {
    reqs_ = std::max(cf.reqs, 0);
    real_ = cf.real;
    cal_ = cf.cal;
    real_cost_ = cf.real_cost;
    for (int k = 0; k < kCtxBuckets; ++k)   // a mark without a cost (a damaged file) is no mark
        for (int t = 1; t <= kMaxT; ++t)
            if ((real_[(size_t) k] >> (t - 1) & 1u) && !(real_cost_[(size_t) k][(size_t) t] > 0.0)) real_[(size_t) k] &= ~(1u << (t - 1));
    apply_table();
    calib_dirty_ = false;
    calib_on_ = false;
    for (int k = 0; k < kCtxBuckets; ++k)
        for (int t = 1; t <= fixed_n_[(size_t) k]; ++t)
            if (!(real_[(size_t) k] >> (t - 1) & 1u)) calib_on_ = true;
    // not calibrated: live costs decide, as without a table, until kCalibReqs requests have collected real rounds
    if (calib_on_ && reqs_ < kCalibReqs && fixed_ && live_saved_) hold_table();
}

void DraftPolicy::start_passive(const CostFile& cf, int n) {
    passive_ = true;
    // every context bucket (upstream builds bucket 0 alone: its cost-aware buckets are opt-in, as XeStrata-890M's are
    // off the HIP integrated GPU); a bucket no request reaches gets no table and takes the nearest one's (fill_bucket)
    real_ = cf.real;
    cal_ = cf.cal;
    real_cost_ = cf.real_cost;
    learned_ = {};
    fixed_n_.fill(std::clamp(n, 1, kMaxT));
    for (int k = 0; k < kCtxBuckets; ++k)
        for (int t = 1; t <= kMaxT; ++t)   // a mark without a cost, or past the table's length, is no mark
            if ((real_[(size_t) k] >> (t - 1) & 1u) && (!(real_cost_[(size_t) k][(size_t) t] > 0.0) || t > fixed_n_[(size_t) k]))
                real_[(size_t) k] &= ~(1u << (t - 1));
    reqs_ = std::max(cf.reqs, 0);
    held_table_ = true;
    fixed_ = false;
    calib_on_ = true;
    calib_dirty_ = false;
}

void DraftPolicy::take_over_passive() {
    const int bk_was = bk_;
    for (int k = 0; k < kCtxBuckets; ++k) {
        bool seen = real_[(size_t) k] != 0u;
        for (int t = 1; t <= kMaxT && !seen; ++t) seen = cost_n_[(size_t) k][(size_t) t] > 0;
        if (!seen) {   // no request reached this context length: no table of its own
            fixed_n_[(size_t) k] = 0;
            real_[(size_t) k] = 0u;
            cal_[(size_t) k] = {};
            continue;
        }
        for (int t = 1; t <= fixed_n_[(size_t) k]; ++t) {
            double v;
            if (real_[(size_t) k] >> (t - 1) & 1u) v = real_cost_[(size_t) k][(size_t) t];         // a real size: its own
            else if (cost_n_[(size_t) k][(size_t) t] > 0 && cost_[(size_t) k][(size_t) t] > 0.0) v = cost_[(size_t) k][(size_t) t];   // live now
            else {
                bk_ = k;
                v = cost_ms(t);   // never measured: the policy's own prior, as without a table
                bk_ = bk_was;
            }
            if (!(v > 0.0)) v = t > 1 ? learned_[(size_t) k][(size_t) t - 1] : 1.0;
            learned_[(size_t) k][(size_t) t] = v;
        }
        monotone_prices(learned_[(size_t) k].data(), fixed_n_[(size_t) k], real_[(size_t) k]);
    }
    passive_ = false;
}

void DraftPolicy::hold_table() {
    for (int k = 0; k < kCtxBuckets; ++k) {
        if (fixed_n_[(size_t) k] <= 0) continue;
        cost_[(size_t) k] = live_cost_[(size_t) k];
        cost_n_[(size_t) k] = live_n_[(size_t) k];
    }
    fixed_ = false;
    held_table_ = true;
}

void DraftPolicy::use_table() {
    fixed_ = true;
    apply_table();
    for (int k = 0; k < kCtxBuckets; ++k) {   // a bucket without a table: emptied, it takes the nearest one's
        const int n = fixed_n_[(size_t) k];
        for (int t = 1; t <= kMaxT; ++t) {
            const bool in = t <= n && cost_[(size_t) k][(size_t) t] > 0.0;
            if (!in) cost_[(size_t) k][(size_t) t] = 0.0;
            cost_n_[(size_t) k][(size_t) t] = in ? 1e6 : 0.0;
        }
    }
    held_table_ = false;
    fill_bucket(bk_);
}

void DraftPolicy::calib_add(int bucket, int t, double ms) {
    static_assert(kCalibN <= CalAcc::kVals, "a size's rounds are all kept for the median");
    if (bucket < 0 || bucket >= kCtxBuckets || t < 1 || t > fixed_n_[(size_t) bucket]) return;
    if (real_[(size_t) bucket] >> (t - 1) & 1u) return;
    // the table is what the rounds of the first seconds of a busy run cost
    if (std::chrono::duration<double>(std::chrono::steady_clock::now() - busy_since_).count() > kFreshS) return;
    double base = learned_[(size_t) bucket][(size_t) t];
    if (!(base > 0.0) && passive_) base = cost_n_[(size_t) bucket][(size_t) t] > 0 ? cost_[(size_t) bucket][(size_t) t] : 0.0;
    CalAcc& a = cal_[(size_t) bucket][(size_t) t];
    if (a.n >= kCalibN) return;
    if (!(base > 0.0)) return;   // the first round of a size captures its window: no reference to clamp it to yet
    const double r = std::clamp(ms, kCalibLo * base, kCalibHi * base);   // one stalled round moves the price by little
    a.sum += r;
    a.vals[(size_t) a.nv++] = r;
    a.n += 1;
    calib_dirty_ = true;
}

bool DraftPolicy::calibrate(CostFile& cf, std::string& report) {
    report.clear();
    if (!calib_on_ && !calib_dirty_) return false;
    if (held_table_) ++reqs_;   // a request that ran while the table is held back
    char b[96];
    bool changed = calib_dirty_ || held_table_, replaced = false;
    for (int k = 0; k < kCtxBuckets; ++k)
        for (int t = 1; t <= fixed_n_[(size_t) k]; ++t) {
            if (real_[(size_t) k] >> (t - 1) & 1u) continue;
            CalAcc& a = cal_[(size_t) k][(size_t) t];
            if (a.n < kCalibN) continue;
            std::array<double, CalAcc::kVals> v = a.vals;
            std::sort(v.begin(), v.begin() + a.nv);
            const double med = a.nv % 2 == 1 ? v[(size_t) a.nv / 2] : 0.5 * (v[(size_t) a.nv / 2 - 1] + v[(size_t) a.nv / 2]);
            const double base = learned_[(size_t) k][(size_t) t];
            if (!(base > 0.0) || (med >= kCalibOkLo * base && med <= kCalibOkHi * base)) {
                if (base > 0.0) std::snprintf(b, sizeof b, "%s%d rows %.2f -> %.2f", report.empty() ? "" : ", ", t, base, med);
                else std::snprintf(b, sizeof b, "%s%d rows real %.2f", report.empty() ? "" : ", ", t, med);
                report += b;
                real_cost_[(size_t) k][(size_t) t] = med;
                real_[(size_t) k] |= 1u << (t - 1);
                replaced = true;
            }
            a = CalAcc{};   // priced, or disturbed (collected again)
        }
    if (replaced && fixed_) {   // the table is in use: the sizes not reached follow the real ones
        const auto before = cost_;
        apply_table();
        for (int k = 0; k < kCtxBuckets; ++k)
            for (int t = 1; t <= fixed_n_[(size_t) k]; ++t)
                if (!(real_[(size_t) k] >> (t - 1) & 1u) && std::fabs(cost_[(size_t) k][(size_t) t] - before[(size_t) k][(size_t) t]) > 0.005) {
                    std::snprintf(b, sizeof b, "%s%d rows (not real yet) %.2f -> %.2f", report.empty() ? "" : ", ", t,
                                  before[(size_t) k][(size_t) t], cost_[(size_t) k][(size_t) t]);
                    report += b;
                }
    }
    calib_dirty_ = false;
    calib_on_ = false;
    bool due = true;   // every size of every bucket a request has reached is priced (a passive bucket not reached waits for none)
    for (int k = 0; k < kCtxBuckets; ++k) {
        bool reached = !passive_ || real_[(size_t) k] != 0u;
        for (int t = 1; t <= kMaxT && !reached; ++t) reached = cost_n_[(size_t) k][(size_t) t] > 0;
        for (int t = 1; t <= fixed_n_[(size_t) k]; ++t)
            if (!(real_[(size_t) k] >> (t - 1) & 1u)) {
                calib_on_ = true;
                if (reached) due = false;
            }
    }
    // the last size is priced, or kCalibReqs requests have not reached the rest: the table decides from now on
    if (held_table_ && (due || reqs_ >= kCalibReqs)) {
        if (passive_) take_over_passive();
        use_table();
        changed = true;
        calib_on_ = false;
        for (int k = 0; k < kCtxBuckets; ++k)
            for (int t = 1; t <= fixed_n_[(size_t) k]; ++t)
                if (!(real_[(size_t) k] >> (t - 1) & 1u)) calib_on_ = true;
        if (calib_on_) report += std::string(report.empty() ? "" : ", ") + "table in use, the sizes real requests did not reach keep a scaled cost";
    }
    for (int k = 0; k < kCtxBuckets; ++k) {   // the file keeps the learned costs, the marks and the rounds so far
        const int n = fixed_n_[(size_t) k];
        cf.n[(size_t) k] = passive_ ? 0 : n;
        cf.table[(size_t) k] = learned_[(size_t) k];
        cf.real[(size_t) k] = real_[(size_t) k];
        cf.cal[(size_t) k] = cal_[(size_t) k];
        cf.real_cost[(size_t) k] = real_cost_[(size_t) k];
    }
    cf.reqs = reqs_;
    cf.fresh = passive_;
    return changed || replaced;
}

// ---------------------------------------------------------------------------------------------------- DraftCosts

std::string DraftCosts::open(DraftPolicy& p, const std::string& path, const std::string& key) {
    p_ = &p;
    path_ = path;
    std::string why;
    const std::string due = " until every window size has " + std::to_string(DraftPolicy::kCalibN) + " rounds of its own or " +
                            std::to_string(DraftPolicy::kCalibReqs) + " requests have run";
    if (!read_cost_file(path, key, file_, why)) {   // no table yet (a new engine, driver, model or setting discards it)
        file_ = CostFile{};
        file_.key = key;
        file_.fresh = true;
        p.start_passive(file_, p.top_size());
        return "draft costs: no table for this machine yet (" + path + ": " + why + "): it is built from the requests, live costs decide" + due;
    }
    if (file_.fresh) {
        p.start_passive(file_, p.top_size());
        return "draft costs: this machine's table is being built from the requests (" + path + ", " + std::to_string(file_.reqs) +
               " requests so far): live costs decide" + due;
    }
    std::string sizes;
    for (int k = 0; k < CostFile::kCtxBuckets; ++k)
        if (file_.n[(size_t) k] > 0) {
            p.set_fixed_costs(file_.table[(size_t) k].data() + 1, file_.n[(size_t) k], k);
            sizes += (sizes.empty() ? "" : ", ") + std::to_string(file_.n[(size_t) k]);
        }
    p.start_calibration(file_);
    if (p.table_held())
        return "draft costs: this machine's table is loaded (" + path + ", window sizes " + sizes + "), not used yet: live costs decide" + due;
    return "draft costs: this machine's table is used (" + path + ", window sizes " + sizes + "): the same request gives the same windows on every start";
}

std::string DraftCosts::request_end() {
    if (p_ == nullptr) return "";
    p_->request_end();
    const bool was_held = p_->table_held();
    std::string report, why;
    if (!p_->calibrate(file_, report)) return "";
    const bool saved = write_cost_file(path_, file_, why);
    std::string out;
    if (!report.empty()) out = "draft costs: " + report;
    if (was_held && !p_->table_held()) out += std::string(out.empty() ? "" : "; ") + "draft costs: the table is used from the next request on";
    if (!saved) out += std::string(out.empty() ? "" : "; ") + "draft costs: " + path_ + " not written (" + why + ")";
    return out;
}

}  // namespace strata::spec
