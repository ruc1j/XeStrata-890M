// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// include/strata/spec/draft_policy.hpp - per verify round: the MTP's window, or a lookup (suffix) window?
//
// The suffix drafter (prompt lookup) proposes the tokens that followed an earlier repeat of the context. Taken
// whenever it proposes more than the MTP, it lost 2-8% on ordinary text: a long lookup window costs much more to
// verify than the MTP's usual 3-4 tokens, and it is only worth that when enough of it is accepted. llama.cpp's
// lookup decoding answers the same problem with confidence thresholds on its n-gram statistics; here the policy
// learns both sides online and compares expected committed tokens per millisecond:
//
//   MTP     E = the mean tokens a window of that size has committed (EMA), at the measured cost of that size
//   lookup  E(k) = 1 + q + q^2 + ... + q^k for k <= the proposal, q = the acceptance rate of lookup drafts whose
//           match was about as long (4 buckets of match length, decayed counts), at the measured cost of k + 1
//
// and takes the lookup window only when its best E/cost beats the MTP's by `margin`. Costs are the measured round
// times per window size (EMA; sizes not seen yet are scaled from seen ones by a prior shape), so the policy adapts
// to the machine and the context length. Its sizes are part of what it verifies: the same row of a 3-token window and
// of a 4-token window got different argmaxes (XeStrata's own products and dense kernels, RX 9060 XT, 2026-11-14), so
// the sizes moving with the measured times move the answer's last bits. A run whose answer must be the same every time
// fixes them (--spec-window min-p --spec-min-p 0) and turns the suffix drafts off (--suffix-draft 0).
//
// Fixed costs (--draft-costs, opt-in; upstream 4079483b6 .. fc66df986): the windows of the cost-aware rule follow the
// measured round times, so two starts whose times differ choose different windows and print different tokens.  A
// table of round costs per window size, built on this machine from the rounds of its own requests and kept in a file,
// replaces them once every size has its own rounds: from then on no round time moves a cost and the same request gives
// the same windows on every start.  Until then live costs decide, exactly as without the table.  The token statistics
// keep learning either way.
#pragma once

#include "strata/kernels/window_limits.hpp"

#include <array>
#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

namespace strata::spec {

/// The real rounds of one window size collected for the table: n rounds, their sum in ms (each clamped to [kCalibLo,
/// kCalibHi] x the reference cost) and the rounds themselves (up to kVals); the size's price is their median (a cold
/// first use, a stall or another process moves a mean, not a median).
struct CalAcc {
    static constexpr int kVals = 16;
    int n = 0;
    double sum = 0.0;
    int nv = 0;                          ///< how many of vals hold a round (== n when every round is kept)
    std::array<double, kVals> vals{};
};

/// The per-machine table file of --draft-costs: text, valid only for the `key` it was built under (the devices, the
/// driver, the engine, the model and the settings that move a round's cost; the engine builds it).
///   xestrata-draft-costs 1
///   key <one line>
///   bucket <0..2> <ms of T1>,<ms of T2>,...   (the table; none while it is being built: "fresh 1")
///   reqs <n>                                  (requests that ran while the table was held back)
///   fresh 1                                   (no table yet: the rounds of the sizes are being collected)
///   real <bucket> <mask>                      (bit t-1: size t is priced by its real rounds)
///   rc <bucket> <size> <ms>                   (the price of a real size)
///   cal <bucket> <size> <n> <sum> <ms of round 1> ... <ms of round n>
///   end
struct CostFile {
    static constexpr int kMaxT = strata::kernels::kWindowMaxT;
    static constexpr int kCtxBuckets = 3;
    std::string key;
    std::array<std::array<double, kMaxT + 1>, kCtxBuckets> table{};   ///< [bucket][size]; 0 = none
    std::array<int, kCtxBuckets> n{};                                 ///< the length of each bucket's table (0 = none)
    int reqs = 0;
    bool fresh = false;
    std::array<unsigned, kCtxBuckets> real{};
    std::array<std::array<CalAcc, kMaxT + 1>, kCtxBuckets> cal{};
    std::array<std::array<double, kMaxT + 1>, kCtxBuckets> real_cost{};
};
/// false (with `why` for the log) when the file is missing, unreadable, malformed or cut short, or was built under
/// another key: the caller then builds a new table.  Never throws.
bool read_cost_file(const std::string& path, const std::string& key, CostFile& out, std::string& why);
/// Writes the file whole or not at all (a temporary file beside it, then renamed over it); makes the folder.
bool write_cost_file(const std::string& path, const CostFile& f, std::string& why);
/// What a table is valid for: `devices` (the engine's description of its devices and driver), the cost-aware rule's
/// setting (XESTRATA_SPEC_COST) and the engine's command line (a file argument by its name and size, so a moved install
/// keeps its table and another model file does not), without the table's own options, then `extra` (the server's).
std::string draft_costs_key(int argc, char** argv, const std::string& devices, const std::string& extra);

class DraftPolicy {
public:
    static constexpr int kMaxT = strata::kernels::kWindowMaxT;   // (a policy made for 8 rows never looks past them)
    static constexpr int kBuckets = 4;
    static constexpr int kCtxBuckets = 3;   // STRATA_SPEC_COST: row costs per context length (< 6K, < 24K, longer)

    /// `source_costs`: the lookup windows' round times in a table of their own (cost_ms(t, true)), apart from the
    /// MTP windows': a lookup window skips the draft layer, so the two differ in cost at the same size, and one EMA
    /// over both priced each by the other's rounds (htbmixbox/Strata-890M ba96856d1).  The gated path's times
    /// (defer_cost / apply_costs) stay in the shared table.  XESTRATA_SPEC_COST=1 (or STRATA_SPEC_COST=1) turns set_robust
    /// on here (opt-in); the engine's callers then set it from spec_cost_default(), which reads the same two names and
    /// is on by default on the HIP integrated GPU.
    explicit DraftPolicy(int max_t, double margin = 0.03, bool source_costs = false);

    // ---- fixed costs (--draft-costs; see the file comment and CostFile)
    /// Fixed costs for `bucket` (round ms of the windows of 1..n rows): they replace whatever was measured, and from now
    /// on no round time changes a cost, no size is probed for its cost and no size goes stale.  A bucket without a table
    /// takes the nearest bucket's at set_context, fixed too.  ms == nullptr or n <= 0: fixed at the built-in shape.
    /// XESTRATA_SPEC_FIXED=1 (+ XESTRATA_SPEC_FIXED_TABLE="T1,T2,...", every bucket) applies it in the constructor.
    void set_fixed_costs(const double* ms, int n, int bucket = 0);
    bool fixed() const { return fixed_; }
    /// The real rounds of a size become its price at kCalibN rounds (their median, clamped to [kCalibLo, kCalibHi] x the
    /// reference: the learned cost, or the live one while the table is being built); a median outside [kCalibOkLo,
    /// kCalibOkHi] x a learned cost means a disturbed stretch, and its rounds are collected again.  Only the first
    /// kFreshS seconds of a busy run count (a GPU that has been busy for long runs slower and unevenly by size, upstream's
    /// M5 Max measurement; a request kIdleS or more after the last one starts a new busy run).  Until every size up to the
    /// table's length is real, or kCalibReqs requests have run, the table is held back and live costs decide.
    static constexpr int kCalibN = 16;
    static constexpr int kCalibReqs = 100;
    static constexpr double kCalibLo = 0.85, kCalibHi = 1.30, kCalibOkLo = 0.86, kCalibOkHi = 1.29;
    static constexpr double kFreshS = 6.0, kIdleS = 0.5;
    /// A table file as read (start_calibration: its learned table is set by set_fixed_costs first) ...
    void start_calibration(const CostFile& cf);
    /// ... or a file with no table yet (or none at all: `cf` holds only the key): the table of bucket 0 for windows
    /// 1..n is built from the rounds of the requests, live costs deciding meanwhile; at the take-over a size never
    /// reached is priced between the nearest sizes with real rounds (the end's price past either end), so a wider
    /// window never costs less than a narrower one.
    void start_passive(const CostFile& cf, int n);
    bool passive() const { return passive_; }
    bool calibrating() const { return calib_on_; }
    bool table_held() const { return held_table_; }
    /// The request boundaries: they place each round in its busy run (only its first kFreshS seconds count).
    void request_start();
    void request_end();
    /// At a request's end: the sizes with kCalibN real rounds get their price, the table takes over when it is due, and
    /// `cf` gets the table, the marks and the rounds collected so far; `report` says what changed.  True when `cf`
    /// should be written.
    bool calibrate(CostFile& cf, std::string& report);
    /// The largest window size (the table is built up to it).
    int top_size() const { return max_t_; }

    struct Pick {
        bool lookup = false;
        int t = 1;                      // window size (1 + drafts)
    };
    /// `t_mtp`: the MTP's window; `lookup_k`: the lookup proposal's length (0 = none); `match`: its match length.
    Pick choose(int t_mtp, int lookup_k, int match) const;
    /// After the round: the window it used, the drafts accepted, and the round's time (verify + commit + draft).
    void observe(bool lookup, int t, int accepted, int match, double round_ms);
    /// Gated proposals, inside one generation: a round's committed outcome only, never its time. `t` is the
    /// lookup window chosen before the draws, or the configured MTP window; both are fixed by the history. A
    /// lookup window counts as a probe of its size, so a size whose cost is still guessed is tried kProbes times,
    /// not every round. Round times include the next window's drafts and the rows past the accepted ones, whose
    /// Philox counters the next rounds draw again, so they wait (defer_cost) or go to the live policy (observe).
    void observe_committed(bool lookup, int t, int accepted, int match);
    /// Gated proposals, inside one generation: a round's time, held until the committed position passes `end`,
    /// the last Philox counter of any draw that time depends on (the window's drafts and rows, the next window's
    /// drafts when the time includes them). Until then those counters can be drawn again by the next rounds and the
    /// time (rejected rows' routing, the next draft's gated length) would steer a window by its own draws.
    void defer_cost(int64_t end, int t, double round_ms);
    /// Apply, in order, the held times whose `end` is below `committed` (the next window's first counter), exactly
    /// as observe() prices a round, in the context bucket the round ran in.
    void apply_costs(int64_t committed);

    /// --lookup-chain: how many of `k_avail` lookup tokens to append after an MTP window of `t_mtp` tokens whose
    /// drafts all hold with probability `p_mtp` (estimated from the draft layer's own probabilities): the k with the
    /// best expected tokens per ms, E(t_mtp) + p_mtp (c + c^2 + .. + c^k) at the cost of t_mtp + k, if it beats the MTP
    /// window alone by `margin`; c = the acceptance of chained drafts by match length (their own counts). 0 = none.
    int chain(int t_mtp, double p_mtp, int k_avail, int match) const;
    /// After a chained round: the MTP window's size, the chained tokens, the window's accepted drafts, its time.
    void observe_chain(int t_mtp, int k, int accepted, int match, double round_ms);
    double chain_rate(int match) const;
    double lookup_rate(int match) const;   // current q for a match length
    /// measured or scaled round time of a window of t tokens; `lookup`: a lookup window's, from its own table where
    /// source_costs is on and the size was measured (else the shared table, as for the MTP's)
    double cost_ms(int t, bool lookup = false) const;
    /// A per-card row-cost table (round ms of the windows of 1..n rows, the draft included) counted as `weight` earlier rounds each:
    /// the start of a process that already knows the card (STRATA_SPEC_COST_TABLE; tuning time), replaced by what it then measures.
    void seed_costs(const double* ms, int n, double weight, int bucket = 0);
    /// STRATA_SPEC_COST: the context position of the next round; selects the row-cost table of its length (a row of attention costs
    /// more at 16K than at 1K).  A bucket with no measurement yet starts from the nearest one that has.  A no-op without set_robust(true).
    void set_context(long long pos);
    /// The measured round cost of every window size seen: "T1:39.8(n=120) T2:..." for the log / the next process's table.
    std::string cost_table_text() const;
    /// STRATA_SPEC_COST: a width's first three round times count by their least, later ones move its cost down faster than up
    /// and are cut a quarter above it, and the MTP chain tries a width again whose time has grown old - so one disturbed round
    /// cannot price a width out of use (draft_policy.cpp).  Off: the cost EMA as it always was.
    void set_robust(bool on) { robust_ = on; }

    /// COST-AWARE MTP WINDOW (STRATA_SPEC_COST=1): the MTP chain is drafted one step at a time; after draft j (probability p_j)
    ///   E_j = 1 + q_0 + .. + q_j,  q_i = prod_{k<=i} a(p_k)   a() = the acceptance measured for MTP drafts of that probability
    /// is the expected tokens of the window of j + 2 rows, worth E_j / cost(j + 2) tokens per ms (cost() = the measured round time of
    /// that window size, which includes the draft).  The window is the best of the widths seen; the chain goes on only while one or two
    /// more drafts could still beat it (the cost is not convex: on a card with experts in RAM a window of 3 costs about a window of 2).
    /// It only chooses which rows are verified; the output is unchanged.
    struct ChainState {
        double q = 1.0, e = 1.0;
        double best_v = 0.0;
        int best_rows = 1;                ///< the window to verify: 1 + the drafts kept
        int kept = 0;                     ///< best_rows - 1
        /// The widest window whose every draft was kept at its own step: draft idx joins only when the best width
        /// after keep_draft(idx) takes it and the drafts before it joined.  best_rows can widen after a later draft
        /// (the cost is not convex), and that later draft's probability comes from the draft head run on the earlier
        /// draws: a sampled window (STRATA_SPEC_PROB) of best_rows would offer a draft by its own value, which
        /// rejection sampling does not allow; prefix_rows sees each draft's probability only on the rows after it.
        int prefix_rows = 1;
        double best_e = 1.0, best_c = 0.0;   ///< the best width's expected tokens and round ms (the rate rule)
    };
    /// Draft `idx` (0-based, probability p) was just computed: update `st` and say whether to compute draft idx + 1.
    /// idx == 0 starts a new chain.  `cap` = the widest window (rows).
    bool keep_draft(ChainState& st, int idx, float p, int cap) const;
    /// After a round of the MTP window T whose drafts were p[0..T-1): `accepted` of them held.  Teaches draft_rate().
    void observe_probs(const float* p, int n_window_drafts, int accepted);
    double draft_rate(double p) const;     // acceptance of an MTP draft that came with probability p
    /// Under the min-p window rule (`true`): a size not measured is scaled from every measured size, as upstream does,
    /// instead of from the nearest one below.  The rule sizes the MTP's window by the drafts' probabilities alone, so
    /// the costs only weigh a lookup window against it, and in the low-RAM file tier, whose small windows' times swing
    /// with the SSD, the scaling from below priced the short lookups cheap: on the RX 9060 XT in a 32 GiB cgroup
    /// (IQ3_XXS, 32K, the text followed, 2026-10-24) it took five short lookup windows where upstream took two of six
    /// tokens, and decoded 16.6-17.4 tok/s against 17.2-17.7 with the scaling from every size.
    void set_scale_all(bool on) { scale_all_ = on; }
    /// The tokens a window of t committed before any is observed: a typical acceptance of a draft
    static double prior_tokens(int t) { return 1.0 + 0.7 * (t - 1); }
    /// The MTP's own window, from this round's drafts (the idea of DeepInfer's confidence-scheduled verification,
    /// written anew): a draft is kept while the tokens it is expected to add, at its calibrated acceptance, raise the
    /// window's expected tokens per ms at the measured window costs; a size whose cost is only guessed is tried when
    /// its drafts are likely to hold.  `probs[j]`: draft j's probability under the draft layer (0 past the drafts
    /// made).  Returns the window size, 1..max_t.
    int mtp_window(const float* probs, int max_t) const;
    /// While drafting: with `n` drafts made (probs[0..n-1]), whether one more could still pay, at the mean
    /// acceptance of a draft; false as well when the drafts made already stop paying.
    bool mtp_extend(const float* probs, int n, int max_t) const;
    /// After an MTP window of `t` tokens: its drafts' probabilities and the accepted count, which calibrate the
    /// drafts' probabilities against what the model accepts (decayed, as the lookup counts).
    void observe_mtp_drafts(const float* probs, int t, int accepted);

private:
    static int bucket(int match);
    double mtp_tokens(int t) const;
    bool stale(int t, bool lookup = false) const;               // a size not measured for a long stretch of rounds
    double draft_q(float p) const;         // a draft's acceptance, from its probability and the calibration
    double draft_q_mean() const;           // the mean acceptance of a tested draft
    bool mtp_step_pays(int t, double held, double q, double expected) const;
    /// One round's time into the costs: the table's calibration first (--draft-costs), then the live cost unless a
    /// fixed table prices the sizes (`lookup` with source_costs: the lookup windows' own table).  The time as counted.
    double add_cost(int t, double round_ms, bool lookup = false);
    // fixed costs: see set_fixed_costs and start_calibration
    void calib_add(int bucket, int t, double ms);
    void apply_table();
    void hold_table();
    void use_table();
    void take_over_passive();
    void fill_bucket(int b);   // a bucket with no cost yet starts from the nearest one that has (fixed when that is)

    int max_t_;
    double margin_;
    std::array<std::array<double, kMaxT + 1>, kCtxBuckets> cost_{}, cost_n_{};   // round ms by context bucket and window size
    bool source_costs_ = false;                            // the lookup windows' times apart (below), not in cost_
    std::array<std::array<double, kMaxT + 1>, kCtxBuckets> lookup_cost_{}, lookup_cost_n_{};
    int bk_ = 0;                                           // the current context bucket
    bool scale_all_ = false;
    std::array<double, kMaxT + 1> mtp_tok_{}, mtp_n_{};    // tokens committed by MTP windows of that size
    std::array<double, kBuckets> ok_{}, bad_{};            // lookup drafts accepted / windows cut short, decayed
    std::array<double, kBuckets> cok_{}, cbad_{};          // chained lookup drafts, the same (reached rounds only)
    double rounds_ = 0;                                    // rounds observed so far
    bool robust_ = false;
    double rate_tok_ = 0, rate_ms_ = 0, rate_n_ = 0;       // tokens committed and round ms of the rounds timed (decayed)
    void observe_rate(double tokens, double ms);
    bool rate_rule() const;                                // keep_draft compares by the long-run rate (set_robust)
    double clamp_sample(int t, double ms, bool lookup = false) const;
    void update_cost(double& mean, double& n, double ms) const;   // one round's time into a size's cost
    bool stale_chain(int t) const;                                // keep_draft: a size's time is old enough to try again
    static constexpr int kPBuckets = 8;
    std::array<double, kPBuckets> pa_ok_{}, pa_n_{};       // MTP drafts reached / accepted by probability bucket (decayed)
    std::array<std::array<double, kMaxT + 1>, kCtxBuckets> last_{};   // the round at which each size was last measured
    double dq_ok_ = 0, dq_p_ = 0, dq_n_ = 0;               // MTP drafts tested: accepted, their probabilities, count
    // Gated proposals (observe_committed / defer_cost / apply_costs); all zero or empty on any other path.
    std::array<std::array<double, kMaxT + 1>, kCtxBuckets> tried_{};      // lookup windows of that size run with no time yet
    std::array<std::array<double, kMaxT + 1>, kCtxBuckets> delayed_n_{};  // of cost_n_: times applied by apply_costs (not new probes)
    struct HeldCost { int64_t end; int bucket; int t; double ms; };
    std::vector<HeldCost> held_;                                          // defer_cost, oldest first
    bool fixed_ = false;                                   // set_fixed_costs / use_table: no round time changes a cost
    bool calib_on_ = false, calib_dirty_ = false;
    bool passive_ = false;                                 // start_passive: no learned table before the take-over
    bool held_table_ = false;                              // a table is loaded but not used yet: live costs decide
    bool live_saved_ = false;
    std::array<int, kCtxBuckets> fixed_n_{};               // the length of each bucket's table
    std::array<unsigned, kCtxBuckets> real_{};
    std::array<std::array<CalAcc, kMaxT + 1>, kCtxBuckets> cal_{};
    std::array<std::array<double, kMaxT + 1>, kCtxBuckets> learned_{}, real_cost_{};   // the table as set; real prices
    std::array<std::array<double, kMaxT + 1>, kCtxBuckets> live_cost_{}, live_n_{};    // the live costs when it was set
    int reqs_ = 0;
    bool seen_request_ = false;
    std::chrono::steady_clock::time_point busy_since_{}, last_end_{};
};

/// --draft-costs in an engine's serve loop (both engines): the table file of one DraftPolicy.  open() once the policy
/// is the one the requests use, then request_start() / request_end() around every request.  The texts are for the log.
class DraftCosts {
public:
    std::string open(DraftPolicy& p, const std::string& path, const std::string& key);
    bool on() const { return p_ != nullptr; }
    void request_start() { if (p_ != nullptr) p_->request_start(); }
    /// The request's rounds priced, the table taken over when it is due, the file written: "" when nothing changed.
    std::string request_end();

private:
    DraftPolicy* p_ = nullptr;
    std::string path_;
    CostFile file_;
};



}  // namespace strata::spec
