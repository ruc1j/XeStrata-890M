<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# XeStrata on Intel GPUs

English | [日本語](XE.ja.md)

This document records how XeStrata's engine runs on Intel GPUs and how that was checked: what the engine requires of the GPU, and the ported arithmetic and its validation.
How to use it is in the [README](../README.md) and the [details](DETAILS.md); how to build it, what setup does and the packages it needs, in [BUILD](BUILD.md).
The Japanese version ([XE.ja.md](XE.ja.md)) is the original; this is its translation.

XeStrata xe0.2.8-1 runs Strata's engine on Intel GPUs through Level Zero and SYCL; its version is the upstream version it has integrated and the count of XeStrata's releases on it, from 1, joined by a hyphen (xe0.2.8-1; on an upstream pre-release such as the beta, the deb and rpm packages take "~" for its "-": xe0.2.0-beta-1 is 0.2.0~beta-1, a version before the release 0.2.0-1, which upgrades it; releases before 2026-10-13 added a number from the second one on: xe0.1.40.2.1).
It is ported from Strata 0.1.24 (`3ce2523c2823687de5372be3af58534f56cbf286`) and carries part of the changes up to 0.1.39 (`6f32ec07`)
([Integration through Strata 0.1.38](#integration-through-strata-0138)).
The same SYCL code is compiled with intel/llvm for NVIDIA GPUs as well (the contrib-llvm build, [Build and run](BUILD.md#build-and-run)).
Upstream's CUDA build is retired and its sources are removed.
Upstream Strata (`3ce2523`) keeps them, and each Xe source names the CUDA file it ports in a comment.

It is developed and measured on an Intel Arc Pro B70 (Xe2, 32 GB); the speeds in the records are that card's unless they say otherwise.
On NVIDIA it is checked on an RTX 4070 and an RTX 3070.
The engine chooses its paths from what the GPU reports (AGENTS.md, the first rule).
The prefill path's matrix products use a kernel of their own on Xe2's XMX (xmx_gemm), a joint_matrix product that takes the reported tile shape on the Arc A series' XMX and NVIDIA's tensor cores (mma_gemm), the same product through the WMMA instructions on RDNA3 and RDNA4.
A GPU without matrix engines uses the DP4a paths instead.
The experts' products in int8 (their quantized blocks as they are, iq_mmq) run on Intel's XMX (Xe2, Xe-HPG), NVIDIA's tensor cores (Turing on) and the WMMA of RDNA3 and RDNA4, and take DP4a only on a GPU without matrix engines.
Volta, whose matrix engines have no int8, takes the FP16 path.
Qwen3.8's prompt keeps the FP16 path on a GPU that runs xmx_gemm (faster there on the B70).

A smaller card is checked by making the B70 look like one.
`STRATA_VRAM_LIMIT_MIB` caps the VRAM the engine sees, `STRATA_MAX_ALLOC_MIB` the largest allocation it assumes, and
`STRATA_NO_XMX=1` takes the paths without the matrix engines.
The processor's own graphics (a UHD 770 on the development machine) is a second, real configuration for checks.

## Integration through Strata 0.1.54

Of the 70 commits from upstream Strata 0.2.7.1 to 0.2.8 (the engine's version is 0.1.54), those that touch the engine, serve and setup are carried into XeStrata-890M (Issue #56).
The 13 commits upstream XeStrata (MistVVK/XeStrata) integrated in SYCL form on 2026-10-30 were applied three-way.
Most of upstream's changes in this span are its own `sycl/` port (Battlemage's prompt attention, the pipelined pair of Arc cards), DeepSeek, the app and Terminal, and the Mac, none of which this fork has.

Carried over:

- Engine: the RAM the expert arena needs is waited for up to 20 s before the load. A PLE table the residency policy locked in RAM gives its lock back when the lock is what leaves the automatic conversation cache too little room. Parking an idle batch slot no longer drops the main session's copy with the shared root checkpoint. A CPU without AVX-512 is told that the pack folder without `native_experts.txt`, not the CPU, is what stops the engine.
- serve: tool-call recovery (`tool_call_recovery`) is on by default, `"tool_call_recovery": false` turns it off. Damaged arguments, a stray closing think tag and a missing `</function>` no longer cost the call, and a value that quotes a whole call or repeats a parameter name is read as part of the value.
- setup: when the BIOS gives a big carve-out to an APU this run does not use, setup says what RAM the OS loses and how to change it.

Not carried over:

- upstream's `sycl/` changes (the lane-cell prompt attention, the query batch, the pipelined default of two Battlemage cards): this fork's engine is other code.
- Verify and batch windows of 16 rows as the HIP default: it only matters beyond 8 slots in a `--batch` group.
- The automatic expert cache pricing the allocations after the load: this fork's config names `--expert-cache`, and XeStrata has no such warning.
- The route prior default: this fork never took 0.2.7's automatic 1.5, so only the reason upstream took its default back is in the documents.
- Stop ids limited to one-token end markers: the fork's `stop_token_ids` already does that. The wait for an engine an earlier run left behind depends on `strata_control/procs.py`, so only the engine-side wait went in.
- DeepSeek, the app, Terminal and Control, the Mac, the DGX Spark, the defaults of two GPUs.

Checked: the free build, CTest (passes except the two tests that need a pack), the serve Python tests (852, 12 skipped), and the decode before and after on the 890M (the abliterated IQ2_XS, greedy, three interleaved rounds: 31.42, 31.47 and 31.43 tok/s before, 31.07, 31.41 and 30.38 tok/s after, the same answers).
The contrib-llvm build, the Windows cross build and other GPUs are unverified.

### Changes once set aside for a measured difference within the noise (Issue #57)

From 2026-10-31 a change is no longer rejected or left opt-in only because its measured difference on the 890M was within the noise: one that is faster by its logic goes in by default as long as the answers are the same and it is not clearly slower.
Under that rule `STRATA_SPEC_COST` became the default of the integrated GPU on the HIP backend (the speculative decoding item above), the 16-bit reads of the weights carry the non-temporal hint in the AMD device compile (the weights are read once a window, so the cache keeps the activations and the grids), and the 16-row MMQ kernel of the prefill loads the weights' raw words one step ahead (IQ2_XS at 16 rows 594 -> 575 us in the kernel bench, bit-identical).
Two rows a sub-group for the dense GEMV at five columns or more did not go in: it equals the current default (the IQ4_XS local-memory kernel) and is 10% slower on the 8-column ssm_out.
The 890M's decode (the abliterated IQ2_XS, greedy, three interleaved rounds) was 31.88, 29.40 and 28.47 tok/s after the sync and 31.90, 30.79 and 31.71 tok/s after the review, with the same answers; a 15K-token prompt read 314.3, 313.8 and 311.3 -> 314.9, 314.4 and 314.5 tok/s, the same.

### Upstream XeStrata's opt-in speed ports (Issue #58)

The opt-in speed changes upstream XeStrata carried in SYCL form with Strata 0.2.7 came in under Issue #57's rule.
The conversation cache's background copies (`STRATA_CKPT_ASYNC`, `STRATA_PARK_ASYNC`, `STRATA_CKPT_GATHER`) are the default of `--serve` on an integrated GPU, and the dense weights' dequantization on a side stream (`STRATA_DEQUANT_SIDE`) is the default of the integrated GPU on the HIP backend (a 15K-token prompt read 314.1, 305.8 and 310.8 -> 311.1, 308.1 and 308.6 tok/s, the same answers).
`STRATA_SWIGLU_QUANT` (swiglu and the int8 quantization in one kernel) reads a 15K-token prompt 11% slower on the 890M, and `STRATA_PREFILL_WARM` (the per-layer walk warming the experts' pages) has nothing to warm on the 890M, whose experts are in RAM, so both stay opt-in ([DETAILS](DETAILS.md)).
The 890M's decode (the abliterated IQ2_XS, greedy, three interleaved rounds) was 31.52, 30.87 and 31.66 tok/s before and 31.62, 30.73 and 30.41 tok/s with the new defaults, with the same answers.

### Layer-major prefill and the chunk size, the attentions' work-group bound (Issues #59, #63)

The layer-major prefill upstream XeStrata ported to SYCL (`STRATA_PREFILL_LAYER_MAJOR=1`: a super-chunk read layer by layer, so the experts' transfers shrink by the number of chunks and the routed rows are multiplied in groups) did not go in. On the 890M the phase that is the transfer (gather) is 0.9% of a 15K-token prompt's GPU time and the copy wait is 0, so fewer transfers gain nothing; the gain is in the rows an expert product holds a launch, and `--prefill 8192` gives the same (283-294 -> 324 tok/s in the first round). With 62 GiB of RAM the second round went to swap and 71 tok/s, layer-major places residual rows and group buffers of the same size, and its sum order changes the answers, so the chunk stays at 2,048 ([DETAILS](DETAILS.md#amd-apus)).
MarchRaBBiT/XeStrata's change that passes the RDNA matrix attentions a work-group bound to cut their register spills (dd9c2a66a) has nothing to change here: `STRATA_AMD_REQD_WG` of 2026-10-14 gives this fork's kernels the same bound, and on gfx1150 WmmaKernel<256,12,4> takes 236 VGPRs with no spill and PromptAttnMma 216-236 with none. Adding it leaves the ISA the same.

### Serve fixes from other forks (Issue #66)

Ben1078/Strata's 8b90ab807 came in: with `"effort_position": "end"` a request's final user message that holds nothing but a `<system-reminder>` moves into the system turn right before the answer, and the next turn reuses the conversation's checkpoint ([DETAILS](DETAILS.md#api)).
shoftee/Strata's 2feece509 came in as well: a stop closes the listening socket first, writes the engine STOP before QUIT so it finishes its request, and joins the handler threads last (a busy stop 20 s -> about 1 s in shoftee's measurement). A write to a closed pipe (ValueError) counts as the engine's end too.
Ben1078's e8e0cdc40 (a metadata-only GGUF shard) was already in this fork's gguf_reader.hpp.

### Memory sizing on an integrated GPU and the expert source's bounds (Issue #67)

Three changes of shoftee/Strata came in: the expert cache is sized from `MemAvailable`, not from the figure clamped by the cgroup's cap (48defc12e: the cache's pages are not charged to the cgroup); with `--conversation-cache-mib N` on an integrated GPU's `--serve` the auto expert cache's reserve grows by N (ff5809230); and ArenaExpertSource's prefetch, release and pinned check the layer's upper bound (f6094ff1e: 172K-190K prompts read past the table and stopped the engine). The fatal-signal backtrace did not come in.
Ben1078's bd0e37bb4 (mlock of the HIP expert arena) did not come in: this fork's expert cache is hipHostMalloc memory, which the driver pins, and the mapped experts are page cache, not something mlock applies to.

### Separate cost tables for the lookup and MTP windows, the O_DIRECT fallback (Issue #64)

htbmixbox/Strata-890M's ba96856d1 came in. The speculative decoding's DraftPolicy kept one EMA of the round times of the lookup (prompt-lookup, suffix) windows and the MTP windows, but a lookup window runs no draft layer, so the two differ in cost at the same width and each priced the other's rounds. With `STRATA_SPEC_SOURCE_COST` the lookup windows' times go into a table of their own (per context bucket), and the choice of a lookup window reads that table. A width not measured keeps the shared guess, and the gated path (defer_cost / apply_costs) keeps the shared table. On by default on the integrated GPU on the HIP backend; `STRATA_SPEC_SOURCE_COST=0` goes back to one table.
The 890M's decode (the abliterated IQ2_XS, greedy, three interleaved rounds) was 31.53, 30.38 and 26.66 tok/s before, 30.93, 30.02 and 31.78 tok/s with the new default and 28.86, 30.14 and 28.37 tok/s with `STRATA_SPEC_SOURCE_COST=0`, with the same answers; single requests show thermal dips, and the rounds' differences are within the noise.
The same fork's d3ed6bb4a came in too: on a file system that refuses `O_DIRECT`, DirectFile opens the file again through the page cache ([DETAILS](DETAILS.md#troubleshooting)).

### The file tier with every expert cached, the streamed walk's floor, where a decode's time goes (Issues #69, #71, #72)

01xjw/Strata's 735a40ec3 came in: when every expert is in the GPU cache once it is filled, the io prefetch (and the decode gate) the Linux defaults turned on go off and the routing-aware prefetch (a router product a layer on the CPU) does not start ([DETAILS](DETAILS.md#amd-apus)). The 890M's decode (the abliterated IQ2_XS, greedy, three interleaved rounds) was 31.65, 31.71 and 31.65 tok/s before and 32.35, 32.29 and 30.64 tok/s after (one request of the third round dipped with the heat), with the same answers. The same issue's STRATA_MMAP_PREFETCH (a read-ahead of the experts the CPU pool computes) did not come in, since the CPU computes no expert in this setup; the parallel profile fill at start-up (chenyb999's STRATA_FILL_THREADS) is about the start-up time and is not measured yet.
Of Indigoshores/Strata's (Strix Halo) ideas, the streamed walk from 16 tokens can be tried with `STRATA_STREAM_ALL_MIN` (the default stays 2048): a 523-token prompt read in 1,844 ms either way, so there is nothing to gain. Tiles of three heights exist here already, as the MMQ's choice of kernel by the row count; the PLE prefetch does not apply, since this fork pins the table in RAM.
Where a decode's time goes (`STRATA_DECODE_TIMING=1`, six chats): of a window's 74-87 ms, 61-71 ms wait for the GPU, 5-7 ms are the draft, and the per-layer host work is under 0.2 ms. A decode is one graph a window, so the persistent launches of unimatrix099 and ryan4yin, which assume HIP kernels launched one by one, do not apply.

## Integration through Strata 0.1.53

Of the 796 commits from upstream Strata 0.2.3.1 to 0.2.7.1 (the engine's version is 0.1.53), the 596 that touch the engine, serve, setup and tools layers were looked at, and the changes of 33 commits are carried into XeStrata-890M (Issue #49).
Most of upstream's changes in this span are for the DeepSeek engine, Metal, the DGX Spark and NVIDIA's kernels, which this fork does not have.

The main things carried over:

- serve: when a call for a tool the request did not declare comes back as text, its name is added to the reply as `strata_rejected_tool_calls` and counted in `/metrics/prometheus`. A namespaced tool is also read as `namespace=member`, `namespace_member`, the bare `member` and with every `__` collapsed to `_`.
  `--host ::` listens on IPv6, `/v1/responses` reads `max_output_tokens` 0 and -1 as no cap and carries `timings`, and `repair_latex_escapes` repairs JSON broken by LaTeX escapes before it is checked.
  Chat Completions tool results are put back in call order, a request without messages is a 400, and an image encoder that cannot start gives a 503.
- engine: the n-gram table is kept in RAM when the RAM holds the table, the expert arena and 8 GiB more (`STRATA_PLE_AUTO_RAM=0` turns it off; the 890M's 62 GiB does not qualify).
  A failed direct PLE read no longer leaves the next ticket waiting for ever, and the table is protected when several threads use it.
  `STRATA_RAM_ADAPT` swaps up to four pairs a layer and copies an expert out of its stage buffer instead of reading the file; `--adapt-decay` takes 0 to 1 only; the retained K/V of a restored conversation is never used to park another one.
- setup and tools: cmake and ninja come from the venv first and an old cmake is not used; a build folder configured with another generator is configured afresh.
  A download falls back to the published size when the length cannot be read and says so when no data arrives for two minutes; the pip record remembers which OS wrote it. gguf_reader bounds its header reads, and calibrate goes on when an engine dies during a measurement.

Not carried over:

- The DeepSeek engine (src/dsv41, trellis), Metal and the Mac, the DGX Spark (the arm64 CUDA build and the sm_121 table), NVIDIA's kernels and defaults (the sm_80, sm_89 and sm_120 tables, MMVQ-IL, the fused prompt, qsa_select), the two-card layer-major prefill and pipelined batch, the gfx12 (R9700) defaults, the Windows gfx1151 defaults.
- setup and serve changes that depend on upstream's product layers (app, Terminal, Control, packaging, docker).
- The conversation-cache changes that depend on CUDA copy streams (HIP has none), telling a checkpoint copy's failure by a sticky error, KVA's capture and oracle modes, and upstream's browser tests (tools/webtest).
  The QFUSE commit fix and the q8_1 division fix do not apply: the fork's windows self-commit and all its divisions are IEEE.

Deferred at the sync and carried over on 2026-10-29 in Issues #50 to #55: the DGX Spark setup values (`--mtp-hnorm stream` as the integrated GPU's default), KVA (`STRATA_KVA`), the SPEC_PROB proposal law with the window-invariant CPU rows (#3416), the AVX-512 gather rows and exact rows, the slot park, the checkpoint failure handling and the window graph eviction, and the rework of the web page (history, languages, KaTeX, previews, Load / Unload, MCP pictures).
Checked after the ports: the free build, CTest (passes but for the two tests that need a pack), the serve Python tests, and the decode before and after on the 890M (abliterated IQ2_XS, greedy, three interleaved rounds: 31.91, 31.88 and 31.65 tok/s before the sync, 31.90, 31.88 and 31.97 after it, 30.45, 31.91 and 31.95 after the ports, the same answers). The defaults change neither the speed nor the answers; what is faster is `--mtp-hnorm stream`, now setup's default (+5%), and the opt-in `STRATA_KVA=1` (a 15K-token prompt reads 18% faster).

Checked: the free build, CTest (passes but for the two tests that need a pack), the serve and tools Python tests, and the decode before and after on the 890M (abliterated IQ2_XS, greedy, three interleaved rounds: 30.93 and 31.46 tok/s before, 31.79, 31.46 and 31.44 after, same answers).
The contrib-llvm build, the Windows cross build and other GPUs are unverified.

## Integration through Strata 0.1.48.1

Of the 1,116 commits from upstream Strata 0.1.45 to 0.2.3.1 (`7ddc17a3`; the engine's version is 0.1.48.1), the changes to the engine, serve, setup and tools are carried into XeStrata-890M.
Upstream's 0.2 added the app, the Terminal and the Control layer around the engine; the engine is one part of it.

The main ones:

- Speculative decoding: the cost-aware MTP window, `STRATA_SPEC_COST`. From each draft's probability it computes the expected tokens of every window width and divides by the measured acceptance and the round's time.
  On the 890M (6 chats, the abliterated file, 28 W, `--spec 6`, greedy, 3 interleaved rounds, 2026-10-23) the medians of 3 rounds were 30.9 tok/s with the default (`STRATA_SPEC_CUM`) and 31.5 tok/s with `STRATA_SPEC_COST=1`, within the noise; since the rule stops drafting the widths its cost table says would lose, it became the default of the integrated GPU on the HIP backend on 2026-10-31 (Issue #57; `STRATA_SPEC_COST=0` restores the `STRATA_SPEC_CUM` rule; elsewhere it stays opt-in).
  The speculative loop's start rule is named `spec_loop_runs()`, and `STRATA_SPEC_ADAPT_EVERY` / `STRATA_SPEC_ADAPT_WARM` (opt-in) swap the adaptive tier more often at the start of a run.
  The measurement switches `STRATA_SPEC_TRACE`, `STRATA_SPEC_T_CYCLE`, `STRATA_ROUTE_STATS`, `STRATA_SPEC_WATCH` and `STRATA_BATCH_TELEMETRY` are off by default.
- serve: after non-finite logits in a window that reads the prompt's tail, the next request reads its whole prompt through the batched path (#3465).
  A request whose prompt fits but prompt + max_tokens does not gets max_tokens shortened to the room left instead of a 400 (the default; `--no-fit-max-tokens` keeps the 400).
  The conversation cache is sized from the free RAM (#3493). Serving without an API key is a warning, not a refusal; `/v1/v1/messages` is read as the real path.
  One definition of tok/s; Anthropic usage reports cache_creation_input_tokens; llama-server's thinking_budget_tokens is honoured.
  Drifted tool-call openers, aliases and wrapper boundaries (#3441, #3075, PR #3556); the image-token limit has a default; a failed picture-reader start carries its stderr tail and exit status.
  An engine program that cannot start is a 503, and a finite RLIMIT_RTTIME is warned about (#3238). Four web-page fixes.
- The CPU pool: the default worker count comes from the physical cores, the host's socket, the cgroup CPU quota and the measured cap of 32 (pool_plan.hpp).
  `--pool-cores` (`STRATA_POOL_CORES`, the config's "pool_cores") names the logical processors the workers take, and the start-up line says where they run. The engine's long-lived threads carry names.
- setup and tools: model files come over 16 connections in pieces, every file setup downloads has a fixed SHA-256 and size, and a pinned version that is gone from Hugging Face stops the download (`STRATA_ALLOW_UNPINNED=1` is the opt-out).
  A file with a wrong SHA-256 is repaired piece by piece, and a file of the user's own (`--gguf-dir`) is never deleted. iq_pack continues an interrupted write at the next layer, `--ple-io ram` is chosen on an SSD too when the RAM has room, and the MTP fetch shows progress and retries.

Not taken:

- Upstream's new product layer: the app (apps/), the Terminal and Control (strata_cli/, strata_control/), packaging, Docker, integrations, with their tests and documents. XeStrata has its own setup, serve and deb/rpm packages.
  The serve and setup changes that depend on them (the TLS certificate check, user_args, engine packages, lifeline, cloudflared) are not taken either.
- The choice among several GPUs (gpu_roles, an integrated GPU never the last stage, hip_auto_devices): XeStrata's serve numbers the cards its own way, and this machine has one GPU.
- Gyro (the APR trellis formats TQ2_T, TQK6 and TQK7 and the Hadamard rotation): a large port of CPU and GPU kernels and the pack, with no gain for this fork's goal (IQ2_XS decode).
- The DFlash2 drafter (CUDA kernels and weights; upstream's auto keeps MTP too).
- The Linux file tier (zero-copy, io prefetch, the drive probe, the prefill planner) and the NUMA split of the pool: on this machine every expert is in RAM and there is one socket, so neither can be measured.
- NVIDIA's per-generation kernels (Volta, Pascal, the sm_120 and sm_86 defaults), NVFP4, Metal, upstream's sycl/, Windows packaging.
- HIP-specific work XeStrata has no counterpart for: hipBLASLt tables, HIP virtual memory, the hipBLAS warm-up, the gfx12 and integrated-GPU WMMA defaults (this fork's own kernels are the default), the 8-bit hyper-connection (this fork's `STRATA_HC_Q8` already does it).
  Opening the queues in a fixed order is unmeasured.

Checked: the free build, the C++ tests that changed (spec_start, draft_policy, suffix_drafter, pool_plan, pool_cores, conversation_memory, batch_telemetry), the Python tests of serve and tools, and the CPU build of the picture reader (strata-vision).
The contrib-llvm build, the Windows cross build and other GPUs are unverified.

## Integration through Strata 0.2.2

The changes of upstream Strata 0.2.1.1 to 0.2.2 (`011c9e60`) are carried into Xe.
Most of upstream's changes in this range are for the app, the terminal, the Mac and upstream's `sycl/`, and are not carried in (for the same reasons as 0.2.0-beta).

Carried in:

- The CPU pool: the start-up says where its workers sit, `--pool-cores` (experimental) places them, and the engine's threads carry names; serve passes the setting `pool_cores` to the engine.
- Speculative decoding: the rule for when it starts is gathered in `spec_loop_runs`, and `STRATA_SPEC_WATCH` watches for stalls.
- serve: when a short remaining prefill window gives non-finite logits, the next request reads the whole prompt at once; an engine that cannot start gets a 503 with its reason instead of a dropped connection; `"api_key": "none"` and `--no-api-key` choose no key.
- serve: Anthropic's usage reports the tokens kept for the next request as `cache_creation_input_tokens`; llama.cpp's `thinking_budget_tokens` is read as the thinking limit too; a tool call whose fence is one token off is read as a call, and a call of an undeclared tool comes back as text; the done line's speed comes from the engine's decode time.
- setup: a download whose SHA-256 does not match is not deleted, only the differing pieces are fetched again, and the check of a large file shows its progress; the user's files under `--gguf-dir` are never deleted; free space stops the install only when it cannot fit; folders in the trash are not remembered as an earlier install.
- iq_pack: an experts.bin write that stopped halfway keeps the finished layers and goes on from the next.

Fixed on XeStrata's side during this integration:

- Regressions with intel/llvm v7.2.0: the XMX product and the grouped product lose `joint_matrix_prefetch` (IGC 2.28 cannot handle v7.2.0's prefetch; the B70's 1K prefill goes from 468 to 751 tok/s); AMD's libspirv gains GroupNonUniform's shuffles (the patch `hip-group-shuffle`); iq_mmq declares its work-group size by attribute, which ends the register spills on AMD (the RX 9060 XT's 32K prefill goes from 1,113 to 1,250 tok/s).
- Speculative decoding: the round that builds a window's graph is left out of the measure the window sizes are chosen by; the time of a size not yet measured is estimated from the nearest smaller measured size; Gyro's trellis layers do not cut a window where the VRAM swaps experts (Gyro-M's decode on the B70 at 1K goes from 48.0 to 54.2 tok/s); a window size without a graph shrinks to one that fits when the VRAM is short (the B70 at 262K no longer fails).

Compared with upstream v0.2.2 on the three makers' GPUs (generate with IQ3_XXS and Gyro-M on all three, Q2_0, IQ2_XS, IQ3_S and UD-Q4_K_XL on the B70 and the RTX 4070; IQ3_XXS's serve and everyday use).
In IQ3_XXS's generate, the B70 is 1.8 to 3.0 times as fast as upstream's `sycl/` build in prefill and 12 to 61% faster in decode.
The RTX 4070 is 1.02 to 1.16 times as fast in prefill and 25 to 37% faster in decode; the RX 9060 XT is 2.1 to 2.7 times as fast in prefill and 2 to 7% faster in decode.
In Gyro-M's generate, the B70 is 1.3 to 2.0 times as fast in prefill and 8 to 11% faster in decode; the RX 9060 XT is 1.3 to 2.3 times as fast in prefill and 1 to 2% faster in decode.
The RTX 4070 is 1.08 to 1.12 times as fast in prefill; its decode is even at 32K, and at 1K 3% slower when it follows upstream's text, 17% faster on XeStrata's own text.

Not carried in: the CPU pool split by NUMA node (XeStrata's pool has no groundwork for nodes), the app and the terminal, the Mac, upstream's `sycl/`,
CUDA's Gyro decode with the trellis codebook in shared memory and its Pascal kernels (tried in XeStrata's SYCL kernels: 17 to 19% slower on the RX 9060 XT, and no faster on the B70 or the RTX 4070), and HIP's fixes for gfx11 and gfx1151 alone.

## Integration through Strata 0.2.1.1

The changes of upstream Strata 0.2.0-beta to 0.2.1.1 (`9ae93f7b`) are carried into Xe.
Most of upstream's changes in this range are for the new app, the terminal, Docker and the Mac, and are not carried in (for the same reasons as 0.2.0-beta).

Carried in:

- setup: Meteor Lake's and Arrow Lake-H's LP-E cores count as E-cores; the low-RAM fit counts the draft layer's VRAM (about 1 GB).
- Downloads: 16 connections by default, four neighbouring pieces per request, a 0-0 range asked of a server whose HEAD gives no length; the MTP fetch goes on over a flaky link and stops with setup.
- setup: a new `--data-dir` moves no model files by itself and keeps finding the old place (`--move-data`, `--set-default-data-dir`).
- serve: an engine ended by signal 9 names a finite RLIMIT_RTTIME (PipeWire without rtkit) as the cause.
- Windows: `strata.exe`, `strata-image.exe` and `strata-device.exe` carry a UTF-8 manifest, so paths with letters outside the ANSI code page arrive intact.
- `--batch`: a layer split over two NVIDIA GPUs runs the default as one serial group (+12 to 18% in upstream's measurement).

Compared with upstream v0.2.1.1 on the RX 9060 XT (IQ3_XXS's generate): 2.1 to 2.7 times as fast in prefill, 3 to 7% faster in decode.
Upstream's engine changes do not touch the other GPUs' measured paths (one GPU, the experts in RAM), so the B70 and the RTX 4070 keep the 0.2.0-beta comparison.

Not carried in: the app and the terminal, the Docker distribution, the Mac, upstream's `sycl/`, HIP's integrated-GPU defaults, hipBLASLt tables and the order the GPU queues are opened in (XeStrata's RX 9060 XT prompts stay within 1.5% from start to start: the two speeds upstream fixed do not show), the experts.bin packs' I/O read-ahead and file tier (XeStrata lacks their groundwork), and the Intel GPU picture reader (XeStrata has its Vulkan reader).

## Integration through Strata 0.2.0-beta

The changes of upstream Strata 0.1.45 to 0.2.0-beta (`3d1fb480`) are carried into Xe.
When the release 0.2.0 is out, what it adds is carried in as well.

The main ones:

- Gyro-S and Gyro-M (agentionai): the trellis quantizations (TQ2_T, TQK6, TQK7) and the experts' Hadamard rotation, computed on the CPU (AVX2, AVX-512) and the GPU (decode and prompt), and listed by setup. The pack version is xs2.
- setup: the model files it downloads are pinned by size and SHA-256; large files are downloaded over several connections; the build tools (cmake, ninja) are apart from the run-time requirements; `--ple-io ram` also on an SSD when the RAM has room; pictures on by default for the models that read them.
- serve: a `max_tokens` that does not fit in the context is shortened to the room left instead of answered with 400 (`--no-fit-max-tokens` refuses it as before).
- The CPU pool's default workers are at most 32 and the container's CPU quota; the cgroup RAM check counts the clean page cache as reclaimable.
- The adaptive tier's swapped-in experts are used from the second window by default (`STRATA_ADAPT_LAG=2`, decode +8.6% on the RTX 4070).
- The picture reader (strata-vision) reads only the text model's vocabulary and header (so a file of Gyro's types opens).

Fixed on XeStrata's side meanwhile:

- The windows are cut at the positions of the VRAM experts' swaps only for a model with a layer that the CPU and the GPU round differently (`STRATA_ADAPT_CUT=0|1`). Every layer of Qwen3.8's four files gives the same bits on the CPU and the GPU, so their windows are not cut: on one text, decode 59.3 -> 61.3 tok/s on the RTX 4070 and 80.5 -> 86.9 on the B70.

Compared with upstream v0.2.0-beta on the three makers' GPUs (generate on five model files, serve, everyday use, image generation).
In IQ3_XXS's generate, the B70 is 1.7 to 3.0 times as fast in prefill as upstream's `sycl/` port and 18 to 38% faster in decode;
the RTX 4070 1.02 to 1.14 times in prefill and 23 to 34% faster in decode; the RX 9060 XT 2.1 to 2.7 times in prefill and equal to 8% faster in decode.
Image generation (1024x1024, 8 steps) takes XeStrata 7.6 s on the B70, 13.9 s on the RTX 4070 and 22.5 s on the RX 9060 XT, upstream 52.0 s, 26.6 s and 73.7 s.

Not carried in: upstream's new app (`apps/`) and terminal (`strata_cli`, `strata_control`), the changes to distribution (installers, signed delivery, engine part updates), NVFP4, upstream's `sycl/`,
the CUDA- and HIP-only speed paths (the sm_120 switches and others), `--kv-grow` on HIP's virtual memory (intel/llvm's HIP adapter lacks the virtual memory extension),
the low-RAM form that reads the GGUF in place (experts used without a copy, I/O read-ahead), and upstream's test tools and records.

## Integration through Strata 0.1.45

The changes of upstream Strata 0.1.44.1 to 0.1.45 (`477ac13c`) are carried into Xe.

The main ones:

- The prompt's MoE products read the experts resident in VRAM from their cache slots directly (B70, 1K: prefill +13%). The input is quantized once a token and spread to its k rows.
- The verify window's QSA block scores read a block's keys once for all of the window's rows (RTX 4070, 32K: decode +1.3%).
- The experts' decode keeps the activations in work-group local memory, the default on every GPU it fits (B70: +1%).
- The QSA top-k picks its radix digits with a cooperative scan; a layer whose experts are all resident groups its MMQ experts by their rows, most first.
- A layer split's prefill hands a chunk's rows to the next GPU in eight pieces, so the two GPUs' copies overlap (`STRATA_HANDOFF_PIPE=0`: one copy).
- The conversation cache: a new conversation that starts at another one's mid-conversation checkpoint borrows it (#1164); a conversation that grew little is not parked again (`--conversation-cache-park-gain`); checkpoints are kept spaced back from a conversation's end (`--prompt-cache-spread`).
- The resident RAM mode: with room left in the budget, the experts of the slots the prompt path borrows are kept in RAM too (#1683).
- `--layer-split auto`: when two cards leave a small stage, it picks again by the ratio of their room (#2644).
- serve: the thinking limit (`max_tokens` and Anthropic's `budget_tokens`), the model card's sampling for a request without one, the prompt tokenized from the last shared prefix (#1852), the next turn sent to the slot that holds the conversation's checkpoint (#2518), short prompts admitted first (`STRATA_ADMIT_SHORT_FIRST`, opt-in), parser fixes.
- Carried as opt-in: `STRATA_EXCHANGE_ROTATE`, `STRATA_RAM_ADAPT`, `STRATA_SPLIT_UNIFORM`.

The following were fixed on XeStrata's side during the integration.

- The boundary between the experts' int8 and FP16 products is timed up to 1024 rows an expert. Timed to 256 only, it read IQ3_S's and UD-Q4_K_XL's 32K chunks with the int8 products (B70, IQ3_S, 32K prefill 1,896 -> 2,425 tok/s).
- The choice of the big prompt chunks no longer counts the RAM copy it has already made a second time (B70, UD-Q4_K_XL, 32K prefill 1,760 -> 2,412 tok/s).
- A layer split's later stages no longer hold a flat 1 GiB out of their expert caches, only their verify windows and, on the last stage, the draft head that binds after the cache (B70 + RTX 4070 at 10 GB: the second card's cache +560-700 slots, prefill +4-5%).

Compared with upstream v0.1.45 on the three makers' GPUs (generate on five model files, serve, everyday use, image generation); the B70 is now compared with upstream's `sycl/` port too.
On the RTX 4070, generate's prefill is the same or faster on all five files (IQ3_XXS at 1K and IQ2_XS 17% faster, UD-Q4_K_XL 1%), and decode 13-30% faster.
On the RX 9060 XT, prefill is 2.1-3.2 times as fast, and IQ3_XXS's decode, 2.5-4.7% slower in 0.1.44.1, is now the same to 7% faster.
On the B70, prefill is 1.6-2.5 times as fast as upstream's `sycl/` port and decode 15-95% faster (UD-Q4_K_XL is not compared: the `sycl/` port's answers on it are broken).
Image generation (1024x1024, 8 steps) takes XeStrata 7.6 s on the B70, 13.2 s on the RTX 4070 and 22.3 s on the RX 9060 XT, upstream's sd-cli 52 s, 25.5 s and 73.5 s.
Against the XeStrata before the integration on the same text, decode is as fast on the B70 and the RTX 4070, and the B70's prefill at 1K 35% faster.

Not carried: upstream's `sycl/`, the batch work (more than 8 slots on a layer split, the slots' GPU-to-GPU copies, the admission opt-ins, the one-launch rows),
the Turing (sm_75) speed paths, `--prefill-pipe` (B70 + RTX 4070: prefill 38.6% slower in the smaller chunks), `STRATA_SPLIT_OVERLAP` (XeStrata's stages share no memory word between the GPUs),
the CUDA opt-in paths without a measurement, and upstream's test tools and records.

## Integration through Strata 0.1.44.1

The changes of upstream Strata 0.1.43.1 to 0.1.44.1 (`296b59d2`) are carried into Xe.
From this run on, Windows changes are taken too (except the HIP parts, which XeStrata's Windows build does not have).

The main ones:

- Qwen3.6-35B-A3B (Unsloth's UD-Q4_K_XL, UD-Q6_K_XL and Q8_0 with the MTP layer) is read on the same plain path as
  Ornith-1.5. Upstream gave Qwen3.6 an engine of its own for CUDA; in XeStrata the plain path reads the GDN's F32
  alpha and beta and the MTP layer's BF16 routers. Its answers were compared with llama.cpp's and it runs on the three
  makers' GPUs. Right after a prompt the VRAM tier is filled with the experts that prompt used most (upstream's q36
  idea), which makes decode 20-40% faster (Ornith and llm-jp too).
- Image generation (Qwen-Image-2.1-Turbo, `POST /v1/images/generations`): instead of upstream's stable-diffusion.cpp,
  XeStrata's own SYCL image engine, `strata-image` (the DiT, the VAE and the text encoder; attention on the matrix
  units; resident beside the server, streaming the layers that do not fit in VRAM).
- Unsloth's files: the K-quant experts' weights are decoded once per grouped call (KqSplit, the same bits; B70
  UD-Q4_K_XL decode +7%). setup offers UD-Q5_K_XL, UD-Q6_K_XL and Q8_0 as well (experimental).
- The commit graph does every GDN and QSA layer in one launch (`STRATA_COMMIT_BATCH`, on by default).
- The layer split's automatic placement (among near ties, the one that leaves the tightest card the most room; a slow
  link's miss cost only from a 4x gap), and the VRAM check at start covers the later cards too.
- The 16384-token prompt chunk only where the RAM covers its loan. The arena's MADV_HUGEPAGE is not asked for when
  free 2 MiB blocks cover less than half of it (#2126).
- serve: a long read gives way to a short request at its next chunk (#656), opencode's `<env>` block moved into the
  first user message (`"env_to_user"`, opt-in), the image encoder only while there are pictures (`"elastic"`, opt-in),
  an engine stopped inside the GPU runtime's set-up started again, how a crashed engine ended, the Responses API's
  `reasoning_effort`, pictures matched by their bytes in the conversation cache (#2228), and the stdin reader's QUIT
  race.
- The quantization yardstick (`tools/quant_quality.py`, `tools/quant_formats.py`).
- `STRATA_HC_Q8` (the hyper-connection down/up read in Q8_0; a BF16 pack's quantized at load): both forms are timed
  at load and the faster kept; with Q8_0 on one GPU the BF16 copies leave the VRAM, and on AMD the read also writes
  the next projections' q8_1 (B70 decode +3.5%; an approximation that changes the output).
- Ported and kept opt-in: `STRATA_MTP_PRIO` and `STRATA_COPY_PRIO`, `STRATA_POOL_UMWAIT`, `STRATA_PCIE_MIN1`, `--adapt-async 1` in the arena
  mode, `STRATA_DENSE_F32`.

The following were fixed on XeStrata's side during this integration.

- A failed GPU query is told from an unfinished one, and a wait ends with an error (issue #12). The "RAM is tight"
  warning counts the experts actually placed in RAM (issue #11).
- WSL2: setup not finding the GPU, a failure in the plain path's PCIe split, and a prompt buffer too large (the three
  points of a report).
- A container to install and run it with Podman (podman compose).

Compared with upstream v0.1.44.1 on the three makers' GPUs (generate on five model files, serve, everyday use):
on the RTX 4070 generate's decode is 8-21% faster on all five files and the prefill as fast or faster (IQ3_XXS at 1K
16%, IQ2_XS 15%). On the RX 9060 XT the prefill is 2.1-3.2x, but IQ3_XXS's decode is 2.5-4.7% slower: upstream reads
the hyper-connection down/up projections in Q8_0 by default on gfx12 (+4.4% in upstream, comparing it off and on).
XeStrata times it at load and takes Q8_0 too, but gains only 0.6% on the RX 9060 XT. The B70 is not compared
with upstream; on the same text it is as fast as XeStrata before this integration.

Not ported: the Mac (Metal), upstream's `sycl/`, the NVFP4 experts (on GPUs without FP4 matrix units their prompts
are slower than the IQ formats', and their quality below UD-Q4_K_XL's), upstream's CUDA-only Qwen3.6 engine, gfx1103
in the gfx11 WMMA kernels (held with the RDNA2 and 3 items), `--prefill-head-alias` and the arena's pin by need (they
do not fit XeStrata's USM and arena), and upstream's test tools and records.

## Integration through Strata 0.1.43.1

The changes of upstream Strata 0.1.43 to 0.1.43.1 (`a0116b27`) are carried into Xe, except the Windows implementation.
They were ported one at a time.

The main ones:

- Asynchronous host-to-device copies from pageable RAM go through a pinned slot, or wait until they are sent (upstream #1996).
  They were the draft K/V's tokens, MTP's prompt records and the image embeddings' rows.
- PLE: a slow direct read only warns and no longer switches to mmap (`STRATA_PLE_AUTO_MMAP=1` for 0.1.42's switch).
  `--ple-io mmap`'s prompt reads ask for the next 4096 rows ahead (upstream #2040, #1984, #1971).
- The watchdog: a file tier busy within the last 10 s keeps its I/O allowance. Parking a conversation copies 1 MiB at a time so the watchdog does not stop it,
  and the stall report names an MTP draft wait (upstream #2049, #2065). `STRATA_TRACE=2` waits for the device at each startup mark.
- serve: a refused BGEN slot tries another slot (#1994), a failing before_load hook answers 503 with Retry-After and `/unload?wait=1` is taken (#1997),
  a quoted `</tool_call>` in a message is text (#2026), a call in a Markdown fence inside `<tool_call>` is read (#2037),
  and a client that leaves while its pictures are prepared stops the preparation (#1999).
- Layer split with `--batch`: the default groups no longer give way to `--batch-mtp` (#2038; `STRATA_BATCH_MTP_SERIAL=1` for the old form).
- setup's `--calibrate` tunes the model of `--model` and `--family` (#2036). The benchmark reports only the GPUs the engine uses (#1985, #2062),
  and the server it starts runs without `--expert-profile-save` (#2088).

The following were fixed on XeStrata's side meanwhile.

- Prompt path: the FP16 and int8 experts' rows and groups share one place, so a model with both paths keeps its chunk.
  RTX 4070, UD-Q4_K_XL at 32K: 460 → 617 tok/s (upstream: 620).
- Small VRAM's prefill: the hyper-connection block's rows live in the attention and MoE work's region (`STRATA_PREFILL_HC_IN_REGION=0` for their own place),
  and the FP16 experts' input rows in the upper half of their down outputs' rows (`STRATA_PREFILL_XS_IN_DM=0`).
  IQ3_XXS at 64K on an RTX 4070 limited to 8 GB: 291 → 428 tok/s.
- With `--spec` and an expert cache of no slots, the engine stops before reading the prompt.
  setup warns of a context whose KV cache and expert cache fit in neither VRAM nor RAM, and names a `--context` or `--kv q4_0` that fits.

Measured against upstream v0.1.43.1 on the three makers' GPUs (generate with five model files, the serve path, everyday use).
On the RTX 4070, IQ3_XXS's decode is 11 to 20% faster from 1K to 262K and the prefill equal or faster.
On the RX 9060 XT, the prefill is 2.1 to 3.1 times and the decode 5 to 12% faster.
On an 8 GB card (both engines run with 8 GB of VRAM left), the prefill is 270 against 194 tok/s on the RTX 4070 at 64K and 713 against 306 tok/s on the RX 9060 XT at 32K.

Not carried: the Windows implementation, upstream's `sycl/`, `STRATA_ADAPT_OVERLAP` and its callback thread's tuning (a feature XeStrata does not have),
telling an adaptive stage's landing by a stream-ordered copy instead of a host function (XeStrata raises its flags in a host function only with `--pcie-mode dma`, not on its default path),
a smaller chunk that would lend more slots (#2090; XeStrata's ring is monotonic in the chunk), and upstream's test tools and records (`bench/results`).

## Integration through Strata 0.1.43

The changes of upstream Strata 0.1.42 to 0.1.43 (`7695859a`) are carried into Xe, except the Windows implementation.
They were ported one at a time; each speed idea was A/B-measured on one GPU and taken when it was faster or no slower.

The main ones:

- A guard on the windows whose logits are NaN or infinite: the request ends and the state it would reuse is dropped (upstream #879, on by default; `STRATA_NAN_GUARD=0` turns it off).
- Prompt path: the host copy of the K/V written 4 bytes a lane (`STRATA_KV_WIDE_HOST=0` for the old form; on the B70 at 64K, 1,974 → 2,399 tok/s),
  the expert copies on the GPU's copy-only engine (Level Zero, `STRATA_COPY_ENGINE=0`),
  an expert's gate/up and down dequantized in one launch (`STRATA_PF_DQ_FUSED=0`; +9% on the B70's UD-Q4_K_XL at 32K),
  the chunk's step records and token ids sent from page-locked buffers (upstream #1712).
- Decode: the expert hits folded into the multi-row combine (`STRATA_FUSE_HITS=0`), Q3_K's GEMV unpacking once a block,
  the route-resident swap one sub-group a token (inside the opt-in feature, +3.9% on the B70).
- serve's tail skip off by default when 90% or more of the experts are in VRAM (upstream #1884, `STRATA_TAIL_SKIP_MAX_CACHED`).
- serve's reports (time and tokens of a request that moved between the solo path and a batch slot, `timings.prompt_n`; upstream #1869 and #1870),
  `STRATA_DROP_CACHE_ON_EXIT`, the file-cache hint when RAM is short, `STRATA_TRACE`'s VRAM marks.
- `xestrata --setup --bench-original-compat`, which runs upstream's benchmark mode (`strata_bench.py`) as it measures ([BENCH](BENCH.md)).

The following were fixed on XeStrata's side during this integration.

- The split of the missed experts between the CPU and PCIe refines the CPU's times from the decode ([Splitting the missed experts between the CPU and PCIe](#splitting-the-missed-experts-between-the-cpu-and-pcie)).
- When the RAM cannot hold the whole expert arena, the engine starts as with `--resident-experts` instead of loading it, and moves layers into host USM only while the RAM has room for them (issue #10, [DETAILS](DETAILS.md)).
- An engine that stops on an exception says in one line what it was doing, the SYCL and Unified Runtime codes, the VRAM free and whether the GPU still answers, and the web app shows it (issue #10).
- The web app's GPU readings agree across the three makers (Intel's temperature is the GPU's own, a layer split's other cards, product names, the PCIe path's limit).
- serve writes the checkpoint at the start of the assistant turn into RAM after the reply ([DETAILS](DETAILS.md)). On the RTX 4070 a continuing request waits 222 instead of 288 ms at 1K and 253 instead of 317 ms at 64K.

Measured against upstream v0.1.43 on the three makers' GPUs (generate with five model files, serve, everyday use): XeStrata is equal or faster in every measure.
On the RX 9060 XT, IQ3_XXS decodes 9-13% faster from 1K to 262K (25-45% faster than xe0.1.42) and reads the prompt 2.9-3.5 times as fast.
The RTX 4070's serve follow-ups wait 222 and 253 ms, against upstream's 240 and 282 ms.

Not taken: the Windows implementation, upstream's `sycl/`, the HIP and CUDA kernels' tuning tables and the paths for GPUs there is no machine for (Pascal's rows table, gfx12's fused int8 prompt experts, gfx12's decode HC norm and folded flag waits, HIP layer-split defaults),
the mixed expert profile upstream made for `sycl/` (generate's 32K decode: B70 −2.0%, RTX 4070 +6.2%, RX 9060 XT −5.3%),
what has nothing to act on in XeStrata (`STRATA_ADAPT_OVERLAP`, PcieAuto's share, skipping the GEMM operand staging, refilling lent slots by DMA), and upstream's records (`bench/results`).

## Integration through Strata 0.1.42

The changes of upstream Strata 0.1.40.2 to 0.1.42 (`61b3fb5d`, 0.1.41 included) are carried into Xe, except the Windows implementation.
They were carried one by one by hand.

The main ones:

- On serve's decode, `STRATA_ROUTE_TAIL_SKIP=7` (default): a missed expert a token routes at rank 7 or lower contributes zero to that token (decided for each token, not for the window), and one that every token of a verify window routes that low is neither read nor computed. Upstream keeps it off on HIP; XeStrata measured the KL, a task check and the speed again on the three makers' GPUs and turned it on for all of them.
- `STRATA_ROUTE_RESIDENT` and `STRATA_ROUTE_PRIOR`, which lean a verify window's routing towards the experts in VRAM (both opt-in; they change the answers).
- The prefill CPU share's extension (experts in RAM that is not page-locked as well) and the rule that leaves it off by default when every expert is in VRAM or memory is short.
- `--aux-cpus` (opt-in), `--ple-io direct` falling back to mmap when it measures slow, the stager's RAM profile, the embedding buffer's reuse.
- The memory guard (the cgroup limit and `--memory-limit-mib`), `--conversation-cache-min-tokens`, `--session-save-reclaim`, `logit_bias`, `strata --version`.
- serve's changes (metrics in Prometheus' format, the Monitor's conversation-cache card, AMD GPU readings, the sampling fields, the tool list's order and more) and setup's (how the config is written, `--gguf-dir`, rotational disks, a VRAM reserve on a card that also drives a display, the worker count on hybrid CPUs and more).

Measured against upstream v0.1.42 on the three makers' GPUs (generate with five model files, serve, everyday use):
the B70 and the RTX 4070 match or beat upstream in every cell; the RX 9060 XT does in every cell but generate's decode at
32K (36.9 against 44.6 tok/s) and serve's wait for a follow-up at 1K (526 against 478 ms), and reads prompts 2.2 to 2.4
times as fast at every length. Its decode moves by about a third from run to run in the same setting (upstream's too),
for reasons not yet known.

The following were found during the final comparison and fixed on XeStrata's side.

- When page-locking the RAM copy of the experts is refused as one allocation (the B70's Level Zero takes 30 GiB at most), it is taken in parts cut between experts (`STRATA_PIN_PARTS=0`: as before). UD-Q4_K_XL at 32K on the B70: prefill +17%, decode +14%.
- The measurement behind the CPU/PCIe split of the missed experts holds still from start to start ([Splitting the missed experts between the CPU and PCIe](#splitting-the-missed-experts-between-the-cpu-and-pcie)).
- serve's short reads keep each path's readings with where the read started and choose from the readings near it; a batched read is tried only up to 16K. On the RTX 4070 at 64K the fourth turn's follow-up went from 505 to 304 ms.

Of the paths meant to be faster, those faster on the B70, the RTX 4070 or the RX 9060 XT and slower on none are carried.
The CPU share's auto gate (upstream f2a3b946) slowed 1K prompts on the B70 and the RTX 4070, so it is opt-in (`STRATA_PREFILL_CPU_GATE=1`); the four-way split of the Q8_0 dequantization (upstream e910e410) was faster on neither, so it is left out.
Not carried: the Windows implementation, upstream's `sycl/` (measured against on the same B70), the HIP and CUDA kernels' tuning tables, some paths for GPUs there is no machine for (those for RDNA2 only), and the multi-GPU changes (carried together in work of their own).

## Integration through Strata 0.1.40

The changes of upstream Strata 0.1.39 to 0.1.40.2 (`e8ca9afd`) are carried into Xe, except the Windows implementation.
Of the AMD (HIP) changes, those that apply to XeStrata's AMD path (the same SYCL kernels) are carried: an AMD APU's unified memory (setup, and the engine's free memory of an integrated GPU) and the fused RMSNorm + RoPE, which was meant for every GPU.
Upstream's HIP-only speed switches were measured on and off on an RX 9060 XT (gfx1200); those for gfx11 (RDNA3) and gfx103x (RDNA2) only are not carried, with no such card to try them on.
Upstream's `sycl/` (another port to Intel GPUs) is not taken in: it is measured against on the same B70.
On NVIDIA GPUs the same kernels run in the contrib-llvm build ([BUILD](BUILD.md)).

The main things carried:

- Conversations saved to a file and restored (session files), `ckpt=0` for a request that takes no checkpoint, `pin=N` to keep a shared beginning, and `tools/research_run.py`.
- The PLE table is read in the format the GGUF has it: IQ4_NL (the default table), Q4_0, Q5_0, Q5_1, Q8_0, FP8 (E4M3 with a scale) or BF16.
  Upstream measured the mean per-row error against the BF16 table: Q8_0 0.53%, FP8 2.64%, Q5_1 3.78%, Q5_0 4.25%, IQ4_NL 7.60%, Q4_0 8.55%.
- GPU kernels for the gate/up of native Q4_0, Q4_1, Q5_0 and Q6_K experts; the CPU experts' AVX-2 i-quant and Q2_0 kernels.
- Fused kernels with fewer launches in the verify windows and the prefill path (#783 and others).
- `STRATA_PREFILL_CPU_SHARE=auto` (the default): a short chunk's least-routed experts are computed by the idle CPU pool.
- `--kv-grow`: the K/V takes VRAM only for the cells a request has reached, and the expert cache uses the rest.
  It is on by default where the device's virtual memory mapping granularity is 2 MiB or more (on for the RTX 4070; the B70's is 64 KiB, off).
- `--adapt-async` (the resident mode's default), `--batch-mtp` (the default with `--batch` and `--mtp`), `--pipeline-windows` (opt-in, [MULTIGPU](MULTIGPU.md)) and `--lookup-chain` (opt-in).
- Setup's additions (`--inspect`, `--source modelscope`, a 200K context and more) and the server's changes.

Of the paths meant to be faster, those faster on the B70 or the RTX 4070 without being slower on the other are carried.
One faster on only one of them is chosen at run time from what the device reports, or is opt-in.

The integration builds in the free (intel/llvm), contrib-icpx (icpx) and contrib-llvm (intel/llvm with CUDA) modes, and all 71 CTest cases pass in each (`expert_multi_test` skips on a CPU without AVX-512).
The AVX-512 path passes `expert_multi_test` under Intel SDE (`-icx`).
`expert_parity --selftest` and `pool_test --selftest` need the canonical pack and cannot run under SDE (they pass on AVX2).
On the B70 the icpx and free builds completed all eight answers (thinking on and off), six of them identical.
Prefill 1,148 tok/s, decode after it 98.6 tok/s, short chats 77.3 tok/s (icpx).
The configuration with 8 GiB, 4 GiB, no XMX and six CPU workers, and the RTX 4070, completed the answers as well.
On the UHD 770 the free and icpx builds started and gave a short answer.

## Integration through Strata 0.1.38

Of upstream Strata's changes up to 0.1.38 (`99f3dbd0b21d1401b3769e0c0d963913607f380b`), the single-GPU Linux ones are carried into Xe.
The upstream history is kept as a merge.
The CUDA, HIP and Windows implementations are not carried; of the multi-GPU ones, only the layer split is ([MULTIGPU](MULTIGPU.md)).

Carried:

- `--coupled-draft`: MTP drafts are sampled with the target model's sampling chain (off by default; `--no-coupled-draft` turns it off explicitly).
  This includes the draft vocabulary map and the penalty history.
  A server config can set `coupled_draft`; engine arguments take precedence.
- `--resident-budget-gib`, `--resident-experts` and `--resident-cpu-experts` for the file expert tier.
- `--shared-expert-arena PATH`: completed expert weights shared between Linux processes.
  The data is flushed before completion is published under a file lock.
  The reader verifies the pack's identity before reusing it.
- Native PLE keys accept Q2_0, IQ3_XXS, IQ4_XS and Q8_0.
  A BF16 PLE key in the pack is used when there is one.
- Native RoPE follows the request's scaling configuration.
- MTP downloads verify the pinned checkpoint and refuse invalid ranges.

The paths meant to be faster are not carried: measured on the B70, none was faster.

| Path | On the B70 |
| --- | --- |
| Quantized expert products in the prefill path (`STRATA_PREFILL_MMQ`) | 0.34x of the default XMX FP16 path |
| Fused MoE (`STRATA_PF_FUSED`) | 0.25x of the default path |
| GDN key-head sharing (`STRATA_GDN_KEYHEAD`) | 0.54x of the default path |
| Split hyper-connection read (`STRATA_GR_V3`) | Decode about 18% slower |
| P-core / E-core placement of the CPU pool (`--pool-affinity`) | Decode 1–4% slower |
| AVX2 multi-token K-quant products (`STRATA_KQ256`) | Within the spread |
| Software prefetch in the AVX2 IQ kernels | Within the spread |
| Routing-aware prefetch of the file tier (`STRATA_LOOKAHEAD`) | Decode about 2% slower |

The quantized products and the fused MoE compute on the decode kernels' DP4a dots, without XMX.
A GPU with XMX therefore runs prefill faster on the default path.

The integration builds in both modes, free (dpclang++) and nonfree (icpx), and both pass the 52 CTest cases.
The default configuration's output (IQ2_XS, nonfree) was compared with the pre-integration build (`bae99bd`).
32 tokens from an 18-token chat were the same in both of two runs.
16 tokens from a 4,095-token prompt were the same in all three pre-integration runs.
After the integration, two of three runs matched them and one differed from the first token ([Why outputs change from run to run](#why-outputs-change-from-run-to-run)).
Under Intel SDE (`-icx`), the AVX-512 paths pass `expert_multi_test` and `native_expert_parity --synthetic` (iq3_xxs/iq4_nl, q4_K/q5_1).
`expert_parity` and `pool_test` were skipped: there is no canonical pack.
The integration commit checked that, with the 8 GiB limit, no XMX and two CPU workers, the same 16 tokens come out with and without `--coupled-draft`.
On the UHD 770 the free build answered with a two-token MTP window.
The nonfree build's four-token window stopped inside the Intel runtime.

## Runtime contract

What the engine requires of the GPU and the host, and the rules it keeps.

- BIOS: a discrete card needs Above 4G Decoding and Re-Size BAR enabled and CSM disabled.
  Otherwise the B70's BARs stayed unassigned and the xe driver did not bind.
- Choosing the GPU: the runtime drives one GPU (several with a layer split, [MULTIGPU](MULTIGPU.md#gpu-numbers)): an Intel GPU through Level Zero, an NVIDIA GPU through the CUDA backend (not OpenCL, which lists the same Intel GPUs again).
  It is chosen by what it reports, never by its maker or device ID.
  It needs 32-wide sub-groups, FP16, and device and host USM, and the executable must carry code for it (an NVIDIA GPU with a build without `STRATA_CUDA_ARCHS` is refused).
  `STRATA_GPU_PCI` (setup writes it) names the card by PCI address.
  Without it, a discrete card is taken before the processor's own graphics (Level Zero's integrated flag; the other backends' GPUs count as discrete cards),
  then the one with the most memory, then the most compute units (whose size differs between makers).
  There is no CPU fallback.
  A GPU without what it needs is refused with the reason.
- Queues: one context owns an in-order compute queue and an in-order transfer queue.
  `copy_async` returns an event; `compute_after` inserts the cross-queue dependency.
  For readback, pass the compute completion event to `copy_async` and wait before reading the host buffer.
  The existing parity harness takes this path with separate host USM staging allocations.
  `create_stream` makes further in-order queues on the same context, for the work CUDA put on its own streams.
  `destroy_stream` drains and releases one, and `finish()` drains them all.
  A kernel's `void* stream` is resolved by `Runtime::stream`:
  an engine stream, the compute queue, or null (the compute queue, with synchronous completion where CUDA synchronized); any other pointer is refused.
  Work on different streams is ordered only through events, as with CUDA's non-blocking streams.
- Allocations: device allocations are 4096-byte aligned; bump allocations validate capacity without wrapping.
- The expert arena: `PinnedArena` is one 2 MB-aligned anonymous mapping with `MADV_HUGEPAGE`.
  It is registered with `prepare_for_device_copy` before the loader fills it; a failed mapping or registration throws.
  Host USM is not used for it:
  Level Zero refuses a single USM allocation above `max_mem_alloc_size` (32.5 GB on the B70), and every model's arena but the Coder's is larger.
  The registration is split into pieces no larger than the device's largest allocation, ending at layer starts.
  Past that size the UHD 770 (4 GiB) registered without an error and copied wrong bytes.
  `registered_bytes` is the whole capacity, and `locked_bytes` counts explicit `mlock` bytes (zero).
  Neither `mlock` nor a hugetlbfs pool is used; transparent huge pages back the arena.
  The B70 measurements registered a 47.46 GiB arena under an 8 MiB memlock limit, with `VmLck` 0.
- Following the device: kernels take what the device reports.
  A work-group of 1024 work-items only where the device takes one (the UHD 770 takes 512).
  FP64 only where the device has it; on the UHD 770 and the Arc A series those kernels sum in FP32.
- The verify window's wait: the verify window's GPU waits for the CPU's experts by spinning on a flag in host memory.
  That needs a running kernel to see the host's writes.
  The verifier checks that at start (`doorbell_visible`).
  Where it fails (the UHD 770), each window runs as segments the host launches one after another, with no waiting kernel: the same results, slower.
  `STRATA_VERIFY_SEGMENTED=1` / `0` forces either.
  A window whose verifier layers have every expert in VRAM runs as another graph that plans the experts on the GPU and never waits for the host (upstream cfd3b72; no segments either).
  On the B70 the Coder IQ1_M went from 28.0 to 28.5 tok/s with the same logits.
  `STRATA_VERIFY_RESIDENT_GRAPH=0` turns it off.
- Lifetimes: every source, destination and reader stays alive until its final consumer completes.
  A copy-complete event alone does not allow reuse while a kernel still reads the destination.
  Arena destruction drains every owned queue before freeing USM or releasing the registration.
  USM pointers are used in their allocation context, and their dependencies are the application's responsibility
  (see the [Khronos USM reference](https://github.com/KhronosGroup/SYCL_Reference/blob/main/source/iface/usm_basic_concept.rst)).
- Errors and waits: synchronous SYCL failures propagate to the CLI; asynchronous errors are kept and checked at waits.
  Waits block in the driver; a watchdog thread reports the named wait and ends the process when one lasts 120 s.
  The xe driver resets a job after `job_timeout_ms` (5 s here), so a healthy wait returns well inside that limit.
  A hang or an unsafe teardown failure ends the process without unwinding through live GPU allocations.
  Driver job timeouts are not modified; CTest uses an outer 60-second timeout.

The parity harness also keeps its small embedding-plus-scale capture/replay check through the experimental [oneAPI graph extension](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_oneapi_graph.asciidoc).
This does not validate a whole-token graph or a CPU/GPU doorbell protocol.

## Implemented arithmetic

### Elementwise operations

The Xe library exports `embedding_gather`, `gdn_gate`, `scale_inplace`, `f32_to_f16_bulk`, `silu_inplace` and `rms_norm_weighted`.
The other declarations in `elementwise.hpp` are not ported and not provided by the library.
These kernels and the parity oracle are compiled without fast math and without multiply/add contraction.

All device code is built with `-foffload-fp32-prec-div -foffload-fp32-prec-sqrt`.
CUDA divides and takes square roots with IEEE rounding by default; SYCL device code does not.
`quantize_act_parity` sees the one-ulp difference in Q8_K's `1/iscale`.

| Operation | Reference and unchanged gate |
| --- | --- |
| Packed embedding row | Independent scalar unpacker; separate multiply/add rounding; bitwise equality, signed zero, guards; 108 row cases plus captured executions |
| GDN gate | CPU double softplus with the `x > 20` branch; relative L1 ≤ `1e-6`, negative gates |
| SiLU | CPU double formula; relative L1 ≤ `1e-7`; Xe uses a stable FP32 formula |
| Scaling | CPU FP32 multiplication; bitwise equality |
| FP32 → FP16 | Shared integer conversion; bitwise agreement |
| RMS norm | CPU FP64 sum/mean, epsilon, and optional weight; relative L1 < `1e-6`, finite output, original null-weight check |

No test cases were added; fixture inputs, CPU expected values and tolerances are unchanged.
`quantize_act_parity`, which also carries NumPy-derived FP16 boundary checks, passes on the B70.
New cases require approval.

### i-quants and MMVQ

All of `iq_kernels.hpp` is ported (`src/kernels/xe/iq_kernels.cpp`):
the q8_1 quantizer, dequantization of IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS, Q3_K and Q2_0, their MMVQ,
embedding rows, the prefill path's gate/up expansion, and the grouped native experts.
A warp is a sub-group of 32; its sums use CUDA's butterfly order, and CUDA's byte intrinsics are written out.
SwiGLU in the grouped experts uses the precise exponential where CUDA used `__expf`.

All of `native_mmvq.hpp` is ported too (`src/kernels/xe/native_mmvq.cpp`):
the llama.cpp MMVQ adapters for Q2_0, Q3_K, Q4_K, Q5_K, Q6_K, Q4_0, Q5_0, Q8_0, IQ4_NL and IQ4_XS (1–8 columns, in the exact and upstream layouts), and the dispatcher.
The dispatcher sends the other i-quant types to `iq_mmvq`, as the CUDA one does.
One column also runs the multi-column kernel in the CUDA single-column layout; there are no separate single-column kernels.
The CUDA file kept the two bitwise equal as well.
Calls require an explicit runtime stream and enqueue without waiting, as the CUDA contract required a non-null stream.

The i-quant measurements show all ten formats dequantizing real model rows exactly as gguf-py does (Q2_0: the definition mirrored from ggml).
Their MMVQ is within `iq_parity`'s `2e-2` bound (4.2e-3–5.8e-3), and `iq_parity` reports 0 failures.
Q6_K, Q8_0, Q4_K, Q5_K, Q4_0 and Q5_0 MMVQ are within `2e-2` of FP64 on random blocks.
Columns 2–8 are bitwise equal to single-column calls in the exact layout.

### Other decode kernels

- `src/kernels/xe/rope.cpp`: `rope.hpp`, `native_rope.hpp` and the M-RoPE table of `mrope.hpp` (one pointer: one Xe device)
- `src/kernels/xe/router.cpp`: `router_top10.hpp` and `native_router.hpp`
- `src/kernels/xe/bf16_gemv.cpp`: `bf16_gemv.hpp`, including the ggml mmvf kernels
- `quantize_act.cpp`, `qsa.cpp`, `kv_q8.cpp`, `kv_q4.cpp`, `qsa_decode_attn.cpp`, `kv_stream.cpp`: the headers of the same names.
  The QSA entry points throw `DeviceError` where CUDA exited

The KV host pools (`KvHostPools`) are read and written by kernels, so on Xe they must be host USM.
Memory registered only for device copies is not kernel-addressable.
`native_rope` and `native_router` use precise `pow`/`cos`/`sin`/`exp` where CUDA used fast math.
The decode-kernel checks record `rope_parity`, `router_top10_parity` and `bf16_gemv_parity` passing; all are registered in CTest.

## Models on the B70

Each model went through the same checks (the procedure):

- `native_expert_parity` on layers covering every expert type pairing, and `dequant_bf16_test`
- a short prompt, and 96 greedy tokens against llama.cpp at the pinned commit on the CPU
- the three draft modes with fixed residency
- the server (streaming, a cancel, a three-turn conversation)
- a 26,293-token prompt with and without KV streaming

The tensor inventory pins the files.

| Model | Result | Against llama.cpp: tokens before the first split (the reference's top-two gap there) | Draft modes |
| --- | --- | --- | --- |
| Qwen3.8-Flash-Next IQ2_XS | runs | 96 of 96 without drafts | the same tokens |
| Qwen3.8-Flash-Next IQ3_XXS | runs | 6 (0.090) | differ; see below |
| Qwen3.8-Flash-Next IQ3_S | runs | 61 (0.051) | no drafts = suffix; MTP differs at the same near-tie |
| Qwen3.8-Flash-Next Q2_0 | runs (the native pack; the canonical AVX-512 pack is not run on this CPU) | 47 (0.050) | the same tokens |
| Qwen3.8-Flash-Next Coder IQ1_M | runs | 20 (0.238) | the same tokens |
| Swift 1.5 IQ2_XS | runs | 7 (0.151) | no drafts = suffix; MTP differs |
| Swift 1.5 IQ3_XXS | runs | 62 (0.058, three candidates) | differ (the cache size also differed between runs) |
| Swift 1.5 Q2_0 | runs (note) | 25 (0.273) | no drafts = suffix; MTP differs |
| OrcaRouter IQ3_XXS (compatibility procedure) | **not run**: the repository is gated and needs a Hugging Face login to download | — | — |

Note: in Swift 1.5 Q2_0, layer 13's expert tensors span both shards.
The pack index names a shard per tensor.
Packs from before 2026-10-02 say v4, newer ones XeStrata's own xs1.
xs1 is numbered apart from upstream's v&lt;N&gt;, so neither is taken for the other.

Every split from llama.cpp comes at a near-tie of the reference; after it the comparison says nothing.
The native path writes `--dump-logits` rows from the last prompt token on (its prompt is read batched, without the head).
The records above were taken before that and compare tokens only.

Where the draft modes differ, there are two causes:

- The MTP runs had fewer cached experts (the drafter takes VRAM), so some experts ran on the CPU instead of the GPU.
  From 2026-10-17 the CPU and the GPU (Intel, NVIDIA, AMD) compute the experts whose gate and up rows are i-quants (IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S), IQ4_XS or Q2_0 and whose down rows are IQ4_NL or Q2_0 (every layer of the four Qwen3.8 files) alike, and an expert gives the same bits on either (`canon_math.hpp`, `canon_expert.cpp`).
  The SwiGLU and the int8 scale are built from additions, multiplications and fmas only, the gate/up activations take one scale for each 256 values, and the integer sums of what shares a scale add first, then go into the GPU's lanes with fmas in the GPU's order.
  The answer stays the same when the window, the PCIe split or the tier changes move an expert between the CPU and the GPU.
  K-quant and BF16 experts (Unsloth's files, Ornith, llm-jp) are not covered.
  `STRATA_CANON=0` takes the CPU's former kernels (for comparisons; the GPU's stay, so the CPU's and the GPU's results then differ).
- On Qwen IQ3_XXS, the suffix run differed from the no-draft run even with the same cache.
  The CPU took its multi-token expert kernels (i-quant gate/up and IQ4_NL down) from two tokens on and ggml's single-token dot for one, so a token's rows were summed in a different order when it ran alone than when it ran in a window.
  From 2026-10-16 the CPU computes these and Q4_K, Q5_K and Q6_K with the multi-token kernels from one token on, and Q2_K, Q3_K and Q8_0 with ggml's dot at every token count. A token's rows are the same alone and in a window, and with the same cache the draft modes give bit-identical logits (the GPU side of a multi-token window added no difference before either).
  `STRATA_IQ_MT_MIN=N` takes the multi-token kernels from N tokens as before (for comparisons).
- On an AVX-512 CPU the i-quant grid lookups can go through one gather instruction (a port of upstream 0.2.5). The same words are loaded, so the results are bit for bit the scalar lookups'.
  It is on by default only on the performance cores of a CPU whose CPUID reports AVX-VNNI and hybrid (`cpu_gather_fast`; never a vendor name or a model number), and off on Zen 4 and Zen 5, whose gathers are microcoded and slow. `STRATA_IQ_GATHER=1` / `0` force either.
  With gathers on, an AVX-512 CPU computes one token's gate and up rows of an IQ3_XXS or IQ3_S expert on the exact rows (`iq512x_rows`), bit for bit ggml's single-token dot. The first use checks them against ggml and leaves them out when they differ. `STRATA_NO_IQ512X1=1` turns them off.
  The layers the canonical rows take (this fork's default) use neither.

### Why outputs change from run to run

Which experts run on the CPU follows the cache size, and that changes the output's last bits.
`auto` sizes the cache from the free VRAM, so free VRAM moving between starts moves it by a slot or two and can change the output.
Rounding it down to 64 slots gave every start on a machine the same cache, but it slowed the RTX 4070's decode by 3-4%, so it is not rounded.
Setup's warm-up instead writes the VRAM free at its start, less 32 MiB, into the run config (`--vram-free-mib`): the engine does not count more free at its start than that, so a start with a little more free VRAM gets the same cache (on the B70 it moved by about 20 MiB between starts, 15 slots). A start with less free gets a smaller cache.
When the same output is needed (comparisons across machines or settings too), fix it with `--expert-cache N`.

### The output head's VRAM refusal

Sometimes the upload of the output head is refused VRAM at start.
About 27 GiB is free then, and it is always the first device allocation after the arena is registered for device copies.
The same settings then start normally.
On the development machine it happened 3 times in about 120 starts.

### The VRAM reserve on a small card

On a small card (upstream #496), the default 700 MiB reserve can leave the cache too few slots to lend the prefill path a 256-token chunk.
The engine then lowers the reserve to what leaves that many:
down to 300 MiB, and not below a reserve given with `--vram-reserve-mib`.
When even that is not enough, the start stops and says how many MiB are short and what makes room.
Checked with `STRATA_VRAM_LIMIT_MIB`:
at 4,608 MiB (IQ2_XS, 32K, MTP) the reserve became 482 MiB, leaving the 368 slots it needs;
at 4,096 MiB the start stopped 330 MiB short.

## Splitting the missed experts between the CPU and PCIe

In a verify window the experts missing from VRAM are computed by the CPU or read over PCIe and computed by the GPU.
How many go each way is chosen from times measured at start (after [Project Maya](https://github.com/mw00/project-maya)'s CPU lane; the code is written anew).
Before, it was a share from the link's speed alone (0.55 from 20 GB/s, below that in proportion).

- Measured: the CPU pool's time for an expert (64 taken in turn, past the CPU's cache), the window's copy kernel (`fetch_blobs`) reading one over PCIe,
  the window's expert kernel computing one from VRAM, and the memory's rate as the CPU reads experts.
- Sharing the memory: the CPU and PCIe read the same memory; when their rates together pass the memory's, both are taken as slowed by that factor.
  On the B70 that gives 0.13 ms an expert over PCIe (0.066 alone), as the decode's waits work out.
  The memory's rate is the fastest of five readings 100 ms apart, taken once the kernels' preload thread is done, and in generate before the prompt is read.
  A single reading beside other work came out up to 3x low, and on the RX 9060 XT some starts then gave the CPU every miss, the 64K decode at 21 tok/s (38-41 once steady).
- The split: for each group of a window, the GPU computes its VRAM hits, then (beside them where the window's fetch branch runs) reads and computes the PCIe experts,
  and the CPU computes the rest; the number read over PCIe makes the later side earliest.
  Experts go over PCIe only when that is at least 10% earlier than the CPU taking them all.
- Refined from the decode: the CPU's time at start is measured a token at a time, and the window's pool calls can run slower (1.27-1.29x on the RX 9060 XT, 3.5x beside a busy host).
  The decode's calls are set against the time the split expected for them, and when they are more than 15% off, the CPU's times are scaled by the ratio (after 512 calls with experts on the CPU, then 1,024 and 2,048 more, three updates at most; as upstream's PcieAuto refines its share).
  On the RX 9060 XT, IQ3_XXS decodes at 52.2 instead of 50.3 tok/s at 1K, 51.3 instead of 48.5 at 64K and 44.7 instead of 42.5 at 262K; on the B70 and the RTX 4070, whose times at start match the decode's, nothing changes. `STRATA_PCIE_SPLIT_REFINE=0` keeps the times taken at start.
- The arena: on Level Zero, GPU kernels cannot read the arena registered for device copies.
  Only when the split reads experts over PCIe, the arena moves into host USM a layer at a time (about 7.5 s for IQ3_XXS's 40 GB). `STRATA_ARENA_HOST_USM=1` puts it in host USM from the start.
- The share stays: with `--pcie-frac` (also as `--calibrate` saved it), over a layer split, with `--pcie-mode dma` or `direct`, and when the times cannot be measured.

The RTX 4070 (PCIe 4.0 x4) now computes every miss on the CPU: IQ3_XXS 32K 49.9 -> 57-58.6 tok/s, UD-Q4_K_XL 32K 22.0 -> 25.7.
The B70 keeps every miss on the CPU, as before; the RX 9060 XT reads 1.0 instead of 4.7 experts a layer over PCIe; neither changes speed.

## Images

Images are available only when chosen in setup (off by default).
When they are, the image encoder (`tools/vision`: llama.cpp's mtmd with the model's mmproj) runs on the GPU by default (`./setup.sh --vision yes` or `gpu`).
When it cannot, the CPU encoder takes over; `--vision cpu` uses the CPU encoder alone.
The CPU encoder reads the pictures checked as llama.cpp does, on grids of both shapes.
Every model that runs was checked with images:
two grid shapes, two images in one request, and a conversation that sends the image again and reuses its cache.

The GPU encoder is the Vulkan build (ggml-vulkan, `-DSTRATA_VISION_VULKAN=ON`).
It takes the GPU by its PCI address (`strata-vision --gpu-pci`, from the config's `vision.gpu_pci`):
with Vulkan the first GPU can be the processor's graphics or another card.
On Intel GPUs its driver is Mesa, free software.
It encodes 7–9× faster than the CPU encoder (8 threads); its embeddings differ from the CPU encoder's by 2.4–3.1% (relative L2) and repeat bit for bit.
(The SYCL build of the contrib-icpx mode, ggml-sycl with oneMKL, was removed with that mode on 2026-10-15.)

The server starts the encoders in this order until one is ready:

1. the GPU encoder
1. the CPU encoder (the config's `vision.fallback`)

The console and `/v1/status` (`vision.encoder`, `vision.fallback`) say which one serves and why the ones before it did not start.
When none starts, or the serving encoder stops later, the server keeps serving text and `vision.error` says why.

## The IQ2_XS file's tensors

The frozen model inventory records the Hugging Face revision and the SHA-256 of each complete GGUF header.
The tensor CSV records every tensor's actual type, shape, offset and byte size.
Only headers were fetched for that inventory; a header prefix is not a usable model shard.
Generation with the complete model, the prefill path, MTP and prompt-lookup drafts, the server, and a 26k-token prompt with KV streaming were checked
with IQ2_XS too.
Greedy tokens were compared with llama.cpp on the CPU.

The `IQ2_XS` distribution at the recorded revision holds 1,224 tensors in 12 storage types, with **no tensor stored as IQ2_XS**.
The name describes the quantization budget of a mixed-precision model.

| Tensor role | Observed storage | Required operations and reference |
| --- | --- | --- |
| Routed expert gate and up | IQ2_S in 34 layers, IQ2_XXS in 11, IQ1_M in 3 | Per-format decode, activation quantization, selected rows/GEMV and batched GEMM; pinned ggml `to_float` and CPU FP64 math |
| Routed expert down | Q2_0 in all 48 layers | Native Q2_0 decode/dot and CPU AVX2 path; the same pinned ggml expert oracle |
| Quantized dense/shared expert weights | IQ3_S, IQ4_XS, IQ4_NL, Q6_K, Q8_0, Q2_0 | Per-type decode, dense GEMV/GEMM and SwiGLU; pinned ggml plus FP64 accumulation |
| Token embedding and output head | IQ4_XS | Row decode, projection, sampling; pinned ggml decode and host arithmetic |
| HC/router/SSM/indexer/PLE projections | BF16 | BF16 activation rounding and projection; existing BF16/GR/router/GDN/QSA parity |
| Norms, biases, recurrence parameters | F32; PLE convolution F16 | RMS, gate, convolution, recurrence, indexing; existing host references |
| PLE n-gram table, second shard | IQ4_NL | Integer n-gram hash, selected row reads/decode; existing PLE vectors and actual table rows |

The native expert oracle in `src/kernels/native_expert_parity.cpp` uses ggml dequantization and host FP64 dot products before comparing the CPU and GPU paths.
On the layers of the types above, it also checks that the CPU's rows and the GPU's are the same bits, the tokens at once and one at a time.
Its per-expert relative L1 bound of `3e-2` belongs to that comparison and is not a universal kernel tolerance.
The GPU's q8_1 activation contract and the CPU's activations (`canon_expert.cpp`'s form for the experts of the types above, ggml's `vec_dot_type` for the others) stay explicit.

## Validation inventory

| Tests and assets | Status and limits |
| --- | --- |
| `strata-device --selftest`, `elementwise_parity` | Ported; pass on the B70. The device selftest checks allocation, alignment and capacity, not numerical readback |
| `gguf_reader_test`, `suffix_drafter_test`, `controller_test`, `draft_policy_test`, `conv_cache_test` | Pass with the current toolchain; they do not establish full engine integration |
| `s_gemv_parity`, `s_gemv_q8k_parity` | Ported, registered in CTest, pass on the B70; `--bench` also checks `s_gemv_split`, `s2_gemv_quads` and `s2_gemv_fast` against the naive kernel |
| `dequant_s2_parity`, `s2_gemv_parity`, `s2_gemv_q8_parity` | Ported, registered in CTest, pass on the B70 |
| `rope_parity`, `router_top10_parity`, `bf16_gemv_parity`, `quantize_act_parity`, `qsa_parity`, `kv_q8_parity`, `kv_q4_parity`, `kv_stream_parity`, `gdn_parity`, `gr_parity`, `cvec_parity` | Ported, registered in CTest, pass on the B70 |
| `shared_expert_parity` | Ported, registered in CTest, passes on the B70, including the SYCL-graph replay of the native scalar gate |
| `sampler_parity` | Ported, registered in CTest, passes on the B70 |
| `qsa_prompt_attn_parity` | The existing FP64 reference and FP32 baseline, ported to SYCL (built, not a CTest case: it is also a benchmark). The XMX kernel passes it at int8 and FP16 KV, the tensor-core one at INT8, FP16 and K8V4 (Q4_0 skipped); without the matrix engines the caller keeps `qsa_decode_attn_batch` |
| `native_moe`, `native_gdn`, `native_gdn_preprocess`, `native_ple_postops`, `native_qsa`, `native_qsa_score`, `native_qsa_indexer`, `native_flash_attn`, `qsa_select`, `fused_gdn` (no in-tree parity) | Probes against ggml-cpu at the pinned commit pass on the B70; not registered in CTest |
| `native_expert_parity`, `dequant_bf16_test` | Ported, built with `STRATA_NATIVE_EXPERTS`, not registered (they need the model shards). Pass on the B70 with the shards of every model under [Models](#models-on-the-b70) |
| `iq_parity` | Ported and built, not registered. `tools/iq_fixture.py` writes `logs/iq_fixture` from ranged reads of the pinned model revision. All 10 formats pass. Q6_K/Q8_0 are not in its list |
| `ple_parity` | Has the checked-in `ple_oracle_vectors.inc`; full checks also need the actual table and reference artifacts |
| CPU expert/pool tests and `pool_stress` | Not in this inventory: the canonical expert tests need AVX-512 and/or a pack. The AVX-512 paths run under Intel SDE as AGENTS.md says |
| `serve.test_server`, `serve.test_detok`, `serve.test_mcp` | Pass with the mock engine |
| `tools/test_iq_pack.py`, `tools/test_shards.py`, `tools/test_calibrate.py` | Pass under `python -m unittest` (2026-10-04) |
| `src/ngram/ple_reader_test.cpp`, `src/platform/memory_test.cpp` | Built again; `ple_reader_selftest` passes, `platform_memory_test` fails under the 8 MiB memlock limit. Neither is evidence for the registered arena |

## Upstream settings not remeasured on Xe

These settings were chosen by upstream Strata on CUDA, mostly on a Ryzen 5 7600 with an RTX 5070.
XeStrata uses them as they are; only some have been tuned on Xe.

| Source | Setting | On Xe |
| --- | --- | --- |
| `src/program/generate.cpp` | Native missed-expert PCIe share 0.55 (canonical 0.2), scaled by min(1, link GB/s / 20), the link read as the best of four 256 MiB bursts | On the new development machine 0.15 was 2–4% faster and every larger share slower; the measurement does not support 0.55 |
| `tools/calibrate.py` | Baseline Ryzen 5 7600 + RTX 5070; PCIe candidates 0/0.2/0.35/0.55/0.75, draft floor 0.3/0.5/0.7, worker candidates; interleaved confirmation requires >3% gain | Candidates and baseline unchanged from upstream |
| `setup.py` | `--spec 4`, `--spec-min-p 0.5`, `--expert-cache auto`, `--prefill auto` | `--prefill auto` now picks chunks up to 32768 tokens; a chunk that does not fit with the default ring takes a 96-slot ring (upstream #583; IQ3_S on the B70 limited to 8 GB: 8K and 32K prompts 1.5-2x); one that still does not fit is planned again with the int8 products' group scratch (GU, H, Hq) sized for one group's rows instead of the chunk's (RTX 4070, IQ3_XXS 32K: the chunk 8192 -> 16384, prefill 1,194 -> 2,058 tok/s; a group's tile shape rounds its sums, so the answer can differ a little from the uncapped run's); the window's size comes from the output per ms at the measured window times and the corrected probabilities, not from the `--spec-min-p` floor (`--spec-window auto`, [DETAILS](DETAILS.md#tuning-for-your-pc---calibrate)); the rest as upstream |
| `src/program/generate.cpp` | 700 MiB VRAM reserve (down to 300 on a small card); prefill borrowing and chunk selection; cache adaptation every 4 rounds and up to 96 swaps | As upstream |
| `src/program/generate.cpp` and the CPU pool | Physical-core worker heuristic plus host worker; cache-hit pokes and graph/doorbell scheduling | The worker count was measured on the new machine: 7 to 19 workers decode equally fast, so the default stays |
| `src/kernels/cpu/iq_avx512.cpp` | Software prefetch distance and AVX-512-specific tuning | Not measured: the development machine has no AVX-512 |
