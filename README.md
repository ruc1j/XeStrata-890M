<!--
SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
-->
# XeStrata-890M

English | [日本語](README.ja.md)

**XeStrata-890M is a fork of [XeStrata](https://github.com/MistVVK/XeStrata) specialized for the AMD Radeon 890M
(the integrated GPU of the Ryzen AI 9 HX 370, gfx1150, RDNA3.5) alone.**
It runs the 125-billion-parameter AI model **[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)**
on a single laptop with the Radeon 890M and 64 GB of RAM.
Only this GPU is targeted; this fork does not cover the other GPUs the original XeStrata handles.

Speed on the 890M (Ryzen AI 9 HX 370, 64 GB RAM, Fedora 44, Qwen3.8-Flash-Next IQ2_XS, 28 W APU power)

| | original XeStrata | XeStrata-890M |
| --- | ---: | ---: |
| decode (average of 6 short chats) | 15-22 tok/s | about 31 tok/s |
| prefill 523 tokens | 5.4 s (98 tok/s) | 2.55 s (205 tok/s) |
| prefill 3,481 tokens | 29.1 s (119 tok/s) | 13.1 s (265 tok/s) |
| prefill 2,669 tokens (the same sentence repeated) | 22.0 s (121 tok/s) | 10.0 s (267 tok/s) |

The "original XeStrata" prefill column was measured on this 890M along the original XeStrata's prefill path; the
answers are the same apart from rounding differences.

> **Contents:** [About XeStrata-890M](#about-xestrata-890m) · [What you need](#what-you-need) · [The model](#the-model) ·
> [Install](#install) · [Using it](#using-it) · [When something goes wrong](#when-something-goes-wrong) ·
> [How it works](#how-it-works) · [For developers](#for-developers) · [All the details](docs/DETAILS.md) · [Image generation](docs/IMAGE.md)

## About XeStrata-890M

XeStrata is [Strata](https://github.com/Niko1221/Strata) (for NVIDIA's GPUs) with its GPU computation rewritten in
the open standard SYCL. On AMD GPUs the SYCL code is compiled on intel/llvm's DPC++ HIP target and run on ROCm. This
fork rebuilds that AMD path for the Radeon 890M. The models, the install flow, the app and the API are Strata's and
XeStrata's as they are.

The build is the free build only, of free software only (ROCm is free software). oneMKL and CUDA are not used.
The fork carries in the original XeStrata's changes up to xe0.2.2-1 (Strata 0.2.2) and the engine, serve and setup
changes of upstream Strata 0.2.9 (engine 0.1.55) ([docs/XE.md](docs/XE.md)).

**What an 890M is, and how to use this fork**

- **The CPU and the GPU use the same memory (LPDDR5X).** The pool the GPU can allocate for itself (the GTT) is about
  half of the RAM by kernel default. This fork keeps all 24,576 experts of the model (about 34 GiB) in host memory
  (host USM) and the GPU reads them there directly. The CPU no longer computes experts the GPU does not hold, and
  decode runs on the GPU alone.
- **The matrix engine is WMMA (16 x 16 x 16 f16 and int8).** The distribution's hipBLASLt has no kernels for gfx1150
  and rocBLAS falls back to general kernels that use no matrix instructions. This fork computes the dense matrix
  products with XeStrata's own WMMA kernels (3 times the speed of rocBLAS).
- **Speed is decided by power and heat.** decode's speed follows the APU's power limit (PPT, 28 W by default), not
  its temperature. At 33 W short chats get 3.5% faster, but long generations rise to 92 degrees and get slower, so
  the default stays 28 W. The clock is throttled at 95 degrees, so speed comparisons are done alternating, with the
  GPU cooled first.
- **The CPU's Ryzen AI 9 HX 370 has AVX-512,** and the CPU-side computation takes the AVX-512 paths.

**Main changes for the 890M** (by part, not in commit order)

- The experts' cache in host USM (`STRATA_CACHE_HOST=1`, used together with `--mmap-experts --expert-cache 24576
  --no-prefill-borrow`). On an integrated GPU the CPU is not left spinning while it waits for the GPU, and the file's
  pages are returned to the OS when the host USM cache is filled.
- **decode:**
  - experts' products: the loads of the rows one lane takes are issued together up front (the down of Q2_0, the
    gate/up of IQ2_S, IQ2_XXS and IQ1_M). On AMD, gate/up and SwiGLU are computed in a single kernel, and the form
    that keeps the actives and the lattice table in local memory (LDS) (mode 8) is the default. Weight addresses are
    formed as pointers into the global address space, and the IQ4 lookup is done with byte selection on `v_perm_b32`.
  - the hyper-connection read-out: the fast version (gr_fast) is the default, the BF16 matrices are kept as Q8_0
    (`STRATA_HC_Q8`, default on an integrated GPU), and up computes 4 rows at a time with 8 lanes to a row; the
    read-out is split into three kernels (`STRATA_HC_V2`, default): the normalization writes x = R' w_norm rs once
    into per-stream working buffers, down loads 16 bytes per lane and works 4 rows at a time (2 rows beyond 5
    tokens), reads x without reading the token's arguments, and the workgroup that finishes last sums the partial
    sums of the 20 column blocks into lo, which up only reads (`STRATA_HC_V2=0` for the previous two-kernel form);
    the three are merged into a single launch (`STRATA_HC_FUSE`, default): down's workgroups read their own rows and
    then wait for the end of the normalization, up's workgroups read their own rows and then wait for the end of
    down's sums, so the row loads overlap the previous stage; down loads x once per group of 2 rows and computes the
    current group while reading the next one; the computation is the same and bit-identical (`STRATA_HC_FUSE=0` for
    three launches).
  - the multi-token IQ4_XS GEMV (3 or more columns) shares the actives by loading them into LDS once for the
    workgroup's 8 rows (`STRATA_MMVQ_LDS`, default; each row's computation is the same and bit-identical;
    `STRATA_MMVQ_LDS=0` for the original form).
  - the subgroups' xor exchanges run on DPP and `v_permlanex16`.
  - GDN's step and commit load the token's inputs into LDS first and reduce the per-token barriers.
  - GDN's commit is folded into the next window (`STRATA_GDN_DEFER`, default): commit parks only the adopted rows'
    k, v, gate and beta, and the next window's step hits them while reading the state and writes back once; the
    state (36 layers, about 113 MB) is read and written 3 times per window instead of 2, and the order of the
    computation is the same, so the answers are bit-identical (`STRATA_GDN_DEFER=0` for the original commit).
  - parked rows are kept over several windows and the write-backs are batched (`STRATA_GDN_ACC=K`, default 6 on
    HIP's integrated GPU, 1 elsewhere): commit accumulates the parked rows, a window's step hits all the accumulated
    rows and only then writes back the state of windows holding K or more accumulated rows; a write-back is about
    1.4 ms over 36 layers, but parked rows are hit again in the next window (about 0.18 ms per row), so one
    write-back every two windows balances out. 890M decode is +0.7% at K=6 (35.7 → 36.0 tok/s), -0.3% at 12 and
    -1.0% at 24. The order of the computation is the same, so the answers are bit-identical (`STRATA_GDN_ACC=1` for
    the per-window write-back).
  - the shared expert (the three small GEMVs gate, up and down) runs as a side branch of the graph next to the
    experts' stage (`STRATA_SH_BRANCH`, default on GPUs where a side branch pays off; upstream's counterpart is
    SH_STREAM); the same kernels in the same order, joined before combine, so the answers are bit-identical. 890M
    decode is +2.5-3.0% (35.5 → 36.5 tok/s). `STRATA_SH_BRANCH=0` for the original order.
  - the draft chain and the window length are cut by the product of the drafts' probabilities so far
    (`STRATA_SPEC_CUM=c`, default 0.5 on HIP's integrated GPU, 0 elsewhere for the old rule; not used when
    `--spec-min-p` is 0; on HIP's integrated GPU the cost-aware window `STRATA_SPEC_COST` is the default since
    2026-10-31, and `STRATA_SPEC_COST=0` brings this rule back) rather than each draft's own probability (`--spec-min-p`): draft long while the head is
    confident and cut short once it starts to waver; used with `--spec 6`. On the abliterated build's 6 chats
    (28 W), `--spec 4` with `--spec-min-p 0.5` gave 32.18 / 31.99 tok/s, and `--spec 6` with 0.5 gave 32.76 / 32.67
    (+2.0%); 0.45 gave +2.3% (a comparison from another run), 0.55 gave +1.1%, and 0.4 or below with `--spec 8` are
    slower (a rejected row costs 9.5 ms per row). Keeping `--spec 4`, 0.45 gave +1.3%. Drafts are only proposals, so
    the answers do not change.
  - when an end token (`<|im_end|>` and the like) appears in the draft, the window is cut at that row: the answer
    ends there, so the rows beyond it are checked but never used.
  - sampling requests (temperature > 0) decide the drafts' acceptance by rejection sampling (`STRATA_SPEC_PROB`, a
    port of upstream 33356f72; on by default on HIP's integrated GPU, `STRATA_SPEC_PROB=0` for the old "accept when
    it matches the main sample"): the draft layer draws its drafts from its own distribution q, verify accepts with
    probability min(1, p/q) against the main distribution p, and on a miss redraws from the residual max(0, p − q);
    the output's distribution stays the main model's (checked with `spec_prob_parity`). q is built only from the
    candidates picked by the head's 1-bit sketch, with no penalties (no q changes the output's distribution;
    `STRATA_SPEC_PROB_SKETCH=0` for q from the whole head as upstream does). When it is on by default it is used
    only for requests with temperature ≥ 0.7 (`STRATA_SPEC_MIN_TEMP`) and their windows are capped at 4 tokens
    (`STRATA_SPEC_PROB_MAX_T`; sampling drafts are faster capped at 4, greedy can be stretched to 6). On the
    abliterated build's 6 chats (28 W, temperature 1.0, average over 4 seeds) the old rule gave 28.61 tok/s and
    this 30.54 (+6.7%); the spread between seeds (differences in the answer text) narrows from 26.2-30.3 to
    29.9-31.3. At temperature 0.7 there is no difference.
    The confidence gate on a draft tests the drawn token by default (a port of upstream #3416): the probability mass the
    gate admitted and the rows it refused (NULL) go to verify, and a refused attempt still counts as one proposal.
    `STRATA_SPEC_PROB_GATE=top` restores the earlier form, which tests q's largest probability before the draw. verify
    applies a rule per row kind (a mu offer, a point offer, the NULL tail, a plain bonus row), and a draft token that was
    not in the proposal returns the error token and cuts the window there. The draft policy sets the next window from
    the rows committed and the proposals refused. With the speculative probabilities on, the CPU's expert rows are
    computed in a form that does not depend on the window size (the canonical rows are already, so only
    `STRATA_CANON=0` changes).
  - `STRATA_VERIFY_MARGIN=a,pmin` (opt-in, the answers change): on greedy requests, accept a draft and go on even
    when it is not the main model's argmax, provided the main model's probability for it is at least a times the
    argmax's and at least pmin. On the abliterated build's 6 chats (28 W, `--spec 4`), 0.5,0.3 gave decode +3.3%
    (37 drafts accepted after loosening, 0.005 log-probability let go per token) and 0.3,0.2 gave +6.2% (76,
    0.020). The answer takes a different wording from the point where it was first loosened. The total log-
    probability let go shows in serve's log.
  - the draft layer's head is thinned with a 1-bit sketch (`STRATA_MTP_SKETCH`, default): at startup a sign and a
    mean magnitude are made for each head row, each draft stage picks 4 candidates of every 128 rows with it, and
    only the candidate rows are computed with the original head to choose among. The head reads drop to about a
    quarter of what they were, and the drafts hit with the same probability (about 0.7% fall outside the
    candidates). Drafts are only proposals, so the answers do not change (`STRATA_MTP_SKETCH=0` for the whole head).
  - the Japanese and English draft vocabulary (`data/draft_vocab_ja.bin`, 56,554 tokens): the bytes the draft layer
    reads are about half of the default vocabulary.
  - `STRATA_EXPERT_KEEP=N` (opt-in, the answers change): compute only the N heaviest experts of each token.
  - `STRATA_EXPERT_SER=Kmin,t` (opt-in, the answers change): compute only the Kmin heaviest experts of each token
    and the experts whose weight is at least t times the token's maximum (ik_llama.cpp's SER). 6 chats at 28 W, rt,
    `--spec 4`: 6,0.2 gave decode +1.4% (35.8 → 36.3 tok/s, teacher-forced next token KL 0.056, top-1 92.8%) and
    4,0.4 gave +5.1% (KL 0.218, top-1 83.6%). Compared with `STRATA_EXPERT_KEEP` at the same speed it is better on
    the gentle side (keep 8: +0.8%, KL 0.074) and worse on the strong side (keep 6: +9%, KL 0.18).
  - `STRATA_VERIFY_ECAP=N` (opt-in, the answers change): the experts a speculative window reads are capped to token
    0's full choice plus the N−10 heaviest rows of the drafts (a port of upstream PR #3545). The rows with the cap
    (token 1 and beyond) pick no stop token: logits built without the dropped experts that hit a stop token would
    end the answer early (a bug the PR adds), and the real stop comes at token 0 of the next window. On the 890M
    the windows are short (3 rows on average), so it does not pay off: at 16 / 20 / 26, decode 35.7 / 36.1 / 35.9
    tok/s (35.8 with nothing), teacher-forced KL 0.456 / 0.275 / 0.136 (values from before the guard was added),
    draft acceptance 75% → 69% (16).
- **prefill:**
  - the dense matrix products take whichever is faster at startup, oneMath's library (rocBLAS) or XeStrata's own
    WMMA kernels (HIP GPUs only). `STRATA_BLAS=1` pins the library and `STRATA_NO_BLAS=1` the own kernels.
  - the own WMMA kernels (mma_gemm) give the output's small squares to the workgroups running at the same time in
    an order (blocks of 8 x 8 tiles, `STRATA_MMA_BLOCK`) that keeps the rows they read in the 2 MB L2.
  - the experts' int8 products (MMQ) read the blob in place when all of the layer's experts are in the cache,
    skipping the gather into groups and the copy into the MMQ ring (`STRATA_PREFILL_MMQ_INPLACE=0` restores the old
    form; the experts packed per product: `STRATA_PREFILL_MMQ_GROUP`, default 16). The WMMA version takes the
    output's staging out of local memory and gets 2 workgroups onto each WGP.
  - the dense projections (attn_qkv, attn_gate, ssm_out, attn_q/k/v, attn_output, the shared expert) whose type and
    shape fit (output a multiple of 128) are multiplied with the same int8 product instead of being returned to
    FP16 (T rows to one expert, tiles of 128 rows): prefill of 523 tokens 2.68 → 2.44 s, of 3,481 tokens 13.4 →
    12.8 s. The answers can change by int8 rounding (identical at 523 tokens, 2 words at 3,481).
    `STRATA_PREFILL_DENSE_MMQ=0` returns to FP16. `STRATA_PREFILL_HC_MMQ=1` multiplies the hyper-connection's
    down/up (BF16, output 320) in the same form from a Q8_0 copy, but on the 890M the hc read slows from 1,313 to
    1,397 ms, so it is off by default.
  - on AMD's device compilation the workgroup size each kernel needs is attached to it (`STRATA_AMD_REQD_WG`), and
    the patch to intel/llvm, `hip-reqd-work-group-size`, hands it to the backend's register budget (before it every
    kernel spilled at 128 VGPR). Because of this the dense products take 64 x 128 tiles for the large products of
    long prompts (`STRATA_MMA_SHAPE=5`, e.g. 2048 rows and 2048 columns or more), and MMQ splits the decoder into
    load and decode, issues a step's loads up front, keeps the lattice table in local memory and multiplies the
    scales with a single fma.
  - the host USM of the experts' cache is taken on HIP from the memory the GPU can cache (`hipHostMalloc`'s
    NonCoherent; `STRATA_HOST_NC=0` for the original host USM): the MMQ that reads in place runs about twice as
    fast (prefill of 523 tokens 3.16 -> 2.68 s).
  - required workgroup sizes are attached to QSA's prompt attention (PromptAttnMma: spill 83 → 0, the attention of
    3,474 tokens 1.99 → 1.22 s), GDN's key-head recurrence (908 → 602 ms), and decode's dense GEMV and HC reads as
    well. prompt attention's p.v accumulation is kept in registers in the WMMA accumulator's layout, with the
    staging tiles and the subgroup barriers removed (1.22 → 1.15 s).
  - the prompt's DeltaNet recurrence runs on 16-token chunks on RDNA3's WMMA (`STRATA_GDN_CHUNKED`, default, 64
    tokens or more): preparation kernels make each chunk's T = (I + A)^-1 and P up front, and the scan keeps the
    state in WMMA accumulators per value head. One layer's recurrence and norm go from 14.4 → 8.0 ms at 3,481
    tokens, and prefill is 2-3% shorter. k, q, T, P and the state are rounded to FP16, so the answers can change
    (the average KL of the next token over 20 prompts is 0.020, and the most likely token stays the same).
    `STRATA_GDN_CHUNKED=0` for the FP32 kernels.
  - the own kernels sum in a different order than rocBLAS and MMQ's scale fma rounds differently, so the answers
    can change by rounding differences (`STRATA_BLAS=1` keeps the dense products as they were).

## What you need

| | |
| --- | --- |
| GPU | AMD Radeon 890M (the integrated GPU of the Ryzen AI 9 HX 370, gfx1150). See [AMD GPUs](#amd-gpus) too. |
| CPU | Ryzen AI 9 HX 370 (12 cores, 24 threads, AVX-512). |
| RAM | 64 GB (all of the IQ2_XS experts, about 34 GiB, go in the RAM). [The model](#the-model). |
| Disk | 68 GB for the model, about 6 GB for the MTP layer. An NVMe SSD. |
| OS | Fedora 44 (the distribution's ROCm 7.1.1). |
| BIOS | The memory allotted to the GPU (VRAM) can stay at the default (512 MB). The experts go in host USM, so no larger allotment is needed. Kernel boot options (`ttm.pages_limit` and the like) are not changed either. |

### AMD GPUs

This fork targets only the Radeon 890M (gfx1150).

- Fedora 44's ROCm 7.1.1 rocBLAS has code for gfx1150. hipBLASLt does not, but the dense matrix products are
  computed by the own WMMA kernels, so it does not matter (the `rocblaslt error: Cannot read ...
  TensileLibrary_lazy_gfx1150.dat` in the start log is harmless).
- Ubuntu 26.04's rocBLAS has no code for gfx1150. Use Fedora 44.
- `./setup.sh --check` prints your GPU's gfx after its name.

## The model

The model run and measured in this fork is Qwen3.8-Flash-Next at **IQ2_XS**, the
[compressed GSQ-RCO version by ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
(a 68 GB download). `--expert-cache 24576` puts all the experts in RAM, using about 34 GiB. No other size or
version has been run in this fork. If all the experts cannot go in the RAM, lower `--expert-cache` (decide by the
free memory in the startup log).

## Install

This fork has no packages. It is installed from the source.

1. On Fedora 44, install the packages the build needs and ROCm (`./setup.sh --packages fedora44/amd --license free`
   prints the same one line).

   ```bash
   sudo dnf install python3 git cmake ninja-build gcc-c++ lld hwloc-devel libzstd-devel libzstd-static \
     vulkan-loader-devel glslc spirv-headers-devel mesa-vulkan-drivers blosc2-devel \
     rocm-hip-devel rocm-device-libs rocm-clang rocm-clang-runtime-devel rocblas-devel hipblaslt-devel hipblas-common-devel
   ```

   Make your user able to open `/dev/kfd` and `/dev/dri/renderD*` (`sudo usermod -aG render,video $USER`, then log
   out and back in).
1. Clone the repository and run setup.

   ```bash
   git clone https://github.com/ruc1j/XeStrata-890M.git
   cd XeStrata-890M
   ./setup.sh --license free
   ```

   setup checks the GPU, the RAM and the CPU, asks whether to build intel/llvm (v7.2.0, with ROCm's HIP target)
   here, builds the engine, downloads the model and starts it. Pressing Enter at the questions takes the
   recommended choice.
   - The model and its size: Qwen3.8-Flash-Next at IQ2_XS.
   - The context: how much text it can keep in mind at once. 32,768 is used.
   - Images: whether it should also read pictures.
   - The experimental speed projection: an experimental feature that changes how it answers, off by default. Read
     [what it does](docs/DETAILS.md#experimental-speed-projection-experimental-off-by-default) before you use it.
1. Once it has started, stop it (close its window) and set the configuration for the 890M. Open
   `xestrata-iq2_xs.json`, which setup wrote in XeStrata's folder, set the value of `--expert-cache` in `"args"` to
   `24576`, add `--mmap-experts` and `--no-prefill-borrow`, and add `STRATA_CACHE_HOST` to `"env"`.

   ```json
   "args": [ ..., "--expert-cache", "24576", ..., "--mmap-experts", "--no-prefill-borrow" ],
   "env": { "STRATA_CACHE_HOST": "1" }
   ```

   To use it in Japanese, replace the `draft_vocab.bin` in the folder `--mtp` points at with
   `data/draft_vocab_ja.bin` (the answers do not change and it is a bit faster; setup writes the original file back
   next time, so copy it into a folder under another name, replace it there, and point `--mtp` at that folder).
1. Run `./setup.sh`: it starts, and your browser opens `http://127.0.0.1:8095`.

- **Time:** the first time, the download (68 GB) and building intel/llvm and the engine take a while. If you stop
  it, the next run continues where it stopped. The driver compiles the GPU code the first time it is used, so it
  ends by running the model once to get that done.
- **The PC is heavy while the model starts:** starting the model reads about 34 GiB of experts into RAM. The first
  start takes longest, and the PC can respond slowly for 1-3 minutes. Don't close the window; wait.
- **Where things go:** the model files in `XeStrata-data` next to XeStrata's folder; the model's settings
  (`xestrata-<model>.json`) and the logs in XeStrata's folder.
- **RAM headroom:** the engine counts as the GPU's free memory what Linux can give back minus 6 GiB
  (`STRATA_UMA_HEADROOM_GIB`). On a 64 GB PC that runs few other programs, adding
  `"STRATA_UMA_HEADROOM_GIB": "3"` to `"env"` increases the memory usable for the context.

**Next time**, run `./setup.sh`: it starts right away. Close its window to stop it. The chat in the terminal is
`.venv/bin/python chat.py`.

**Running it while you are logged in:** `systemctl --user enable --now xestrata` keeps the model set up last
running (no browser opens). Set it up with `./setup.sh` once first.

**Updating:** after `git pull`, run `./update.sh`: it updates the engine, the Python packages and the model's
settings (the model does not start). After an update or a re-choosing in `./setup.sh`, check that the `"args"` and
`"env"` you added to `xestrata-iq2_xs.json` are still there.

**Fitting it to your PC:** `./setup.sh --calibrate` measures some of the engine's settings on this PC and keeps the
fastest (about 5-10 minutes).

**Measuring its speed:** `./setup.sh --bench` measures the speed on this PC of the installed model (give it with
`--model IQ2_XS` and the like) and where the time goes, and writes a record to the data folder's
`records/bench-<date-time>/` (about 40 minutes; stop a running XeStrata first). What it measures and what the
record holds: [docs/BENCH.md](docs/BENCH.md).

**Removing it:** `./setup.sh --remove-data` deletes the model files, the settings and the logs (it shows their
sizes and asks), then delete the folder.

Details of the build (the free build, building intel/llvm, what setup does) are in [docs/BUILD.md](docs/BUILD.md).

## Using it

- **In the browser:** `http://127.0.0.1:8095` (it opens by itself when the model starts): **Chat**, the **Monitor**
  of the model and the GPU, CPU and RAM, **About** with the settings and addresses, and **Images** when image
  generation is installed.
- **Chat in the terminal:** `.venv/bin/python chat.py`
- **Apps and coding agents:** add it as an "OpenAI-compatible" provider with the base URL
  **`http://127.0.0.1:8095/v1`**. Any API key and model name work. Apps that use Anthropic's API:
  `http://127.0.0.1:8095/v1/messages`.
- **Thinking:** the model thinks before it answers. The depth is chosen from **Off, Low, Medium or High** in the
  chat page's menu, with `/think low` in `chat.py`, or with your app's "reasoning effort" setting. Off is fastest;
  High suits hard questions.
- **Showing it a picture:** press **Picture** in the chat page; in `chat.py`, type `/image <path>`; in apps, attach
  it.
- **Making a picture:** in the Images tab, or ask in the chat ([Image generation](docs/IMAGE.md)).
- **From a phone or another PC:** run `./setup.sh --host 0.0.0.0 --api-key <secret>` and open the address the
  server window prints ([details](docs/DETAILS.md#streaming-and-connecting)).

**Good to know:** it answers one request at a time (to answer several at once, `"parallel": 2`,
[BATCHING](docs/BATCHING.md)). The first message of a conversation is read in full; after that it keeps the
conversation and reads only what is new, so follow-ups start right away. The API in full, MCP tools and
llama-swap: [DETAILS](docs/DETAILS.md#using-it) and [LLAMA_SWAP](docs/LLAMA_SWAP.md).

## When something goes wrong

**My PC froze or got very slow on the first start.**
It is common while the model starts, most of all the first time: it is loading the experts into RAM. Don't close
the window; wait. Still frozen after 10 minutes? Restart the PC, close other programs (browsers above all) and try
again. If it keeps happening, pick a smaller size (Q2_0 or IQ2_XS).

**It stopped while downloading or installing.**
Run `./setup.sh` again. It continues where it stopped.

**It says it cannot find or use the GPU.**
Check these two.

- That ROCm's libraries (HIP, rocBLAS, hipBLASLt) are installed: the package line of [Install](#install).
- That your user can open `/dev/kfd` and `/dev/dri/renderD*` (the `render` and `video` groups).

**The start log says `rocblaslt error: Cannot read ... TensileLibrary_lazy_gfx1150.dat`.**
It is harmless: hipBLASLt has no kernels for gfx1150, and the dense matrix products are computed by the own WMMA
kernels.

**It says port 8095 is already in use.**
XeStrata is already running: look for its window. If another program uses the port, choose another one, for
example `./setup.sh --port 8081`.

**It is very slow and the disk light stays on.**
The RAM is not enough. Close other programs, lower `--expert-cache`, or pick a smaller size (Q2_0).

**An answer stopped with "the engine stopped unexpectedly".**
Usually too little RAM (Linux stops the engine). Send the same message again: the engine starts again by itself.
If it keeps happening, close other programs or lower `--expert-cache`.

**It gets slower in the middle of a long generation.**
The clock is throttled when the GPU reaches 95 degrees. Don't block the laptop's air intake. Raising the APU's
power limit (PPT) still ends slower on long generations, from the heat (28 W is recommended).

**It says the prompt exceeds the context.**
The conversation is longer than the context you chose. Start a new conversation, or run `./setup.sh` and choose a
longer context.

**Still stuck?**
See the [full troubleshooting table](docs/DETAILS.md#troubleshooting) (for image generation,
[IMAGE](docs/IMAGE.md#when-something-goes-wrong)).
  You can hand an LLM agent that can drive a terminal the [recovery and diagnosis instructions](docs/DIAGNOSE.md)
  and the symptoms.
Open an [issue](https://github.com/ruc1j/XeStrata-890M/issues) and attach XeStrata's folder's
`xestrata-<model>.log`.

## How it works

Models like this one normally run on servers with hundreds of gigabytes of GPU memory. On the 890M the CPU and the
GPU share one RAM, so **XeStrata-890M uses the whole RAM as the GPU's memory**.

- **The model is a team of 24,576 small specialists ("experts"),** and only 10 of them are used to write a word.
  This fork keeps all the experts (about 34 GiB at IQ2_XS) in host memory and the GPU reads them directly. The
  GPU's pool (the GTT) holds the layers every word needs, the KV cache and the working memory.
- **The GPU** takes all the computation every word needs and the experts' computation too. On decode the CPU hardly
  works.
- **The SSD** holds a large table, of which only a few rows are read per word.
- **Guess, then check:** a small helper built into the model (the MTP layer) guesses the next few words, and the
  big model checks them all at once. It keeps the ones that are right and writes the next word itself. The big
  model always decides, so the quality of the answer does not change.
- **Long texts are read in large pieces** (up to 32,768 tokens at a time). prefill's dense matrix products run on
  the 890M's WMMA, and the experts' products on int8 WMMA.

Each part is explained in [docs/DETAILS.md](docs/DETAILS.md#how-it-works) (the original XeStrata's explanation).

## For developers

Before sending a report or a pull request, read [CONTRIBUTING.md](CONTRIBUTING.md) (no dependence on the model number, the licenses, the additional
permission under GPL section 7, what goes in, and how to report).

- **Building:** the free build only (`build/free`, `-DSTRATA_LICENSE=free`, intel/llvm v7.2.0 built with HIP by
  `tools/intel_llvm_build.py`). The steps are in [docs/BUILD.md](docs/BUILD.md). This machine's facts (paths,
  settings, tools) are written in `LOCAL.md` (not in git).
- **Measuring a kernel alone:** `build/free/gemm_bench` (dense products), `iq_mmq_parity --bench` (the experts'
  int8 products), `expert_bench`, `gdn_bench`, `gr_bench`, `mmvq_bench`. Stand-alone runs are measured at the high
  clock (`dpm high`) without pinning it, and A/B runs are done alternating under the same conditions with the GPU
  cooled first (heat moves the numbers a few %). `STRATA_PREFILL_TIMING=1` logs the per-stage GPU times of prefill.
- **Checking the ISA:** emit a kernel's bitcode with `-fsycl-device-only`, turn it into assembly with intel/llvm's
  `llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1150`, and read the metadata's `.vgpr_count` and `.vgpr_spill_count`.
  The AMD backend decides a kernel's register budget from its required workgroup size (intel/llvm's patch
  `hip-reqd-work-group-size`), so kernels are given `STRATA_AMD_REQD_WG`; kernels without it are held to 128 VGPR
  for 1024 work-items and spill.
- **What to keep:** [AGENTS.md](AGENTS.md). The branches for the 890M are chosen from what the hardware reports
  (an integrated GPU, the HIP backend, whether WMMA is present) or an environment-variable opt-in, not from the
  GPU's name. The lints (`tools/lint/run.sh`), the tests (CTest, `python -m unittest`) and the license notices are
  there too. Measurement records are not put in git.
- **The README and docs:** the Japanese ones (README.ja.md, docs/*.ja.md) are written first, and the English ones
  (README.md, docs/*.md) are their translations. `docs/` is the original XeStrata's.

## Credits

- **The original software:** [XeStrata](https://github.com/MistVVK/XeStrata), by MistVVK and the XeStrata
  contributors, and its own original [Strata](https://github.com/Niko1221/Strata), by Niko1221 and the Strata
  contributors.
- **The models:**
   - [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), by the Qwen team.
   - the compressed version (IQ2_XS) by [ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
     and the [Coder](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF).
   - [Swift 1.5](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF), by UkisAI.
   - UD-IQ4_XS, UD-Q4_K_XL, UD-Q5_K_XL, UD-Q6_K_XL and Q8_0, by [Unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF).
   - [Gyro](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF), by agentionai.
   - the [abliterated versions](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF), by huihui-ai.
   - [Ornith-1.5](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF) by Ornith AI (the GSQ-RCO ones by pfeifferj), [llm-jp-4.1](https://huggingface.co/llm-jp/llm-jp-4.1-32b-a3b-thinking-gguf) by LLM-jp,
     [Qwen3.6-35B-A3B](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF) by Qwen and [GLM-5.3-Flash](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF) by Z.ai (both GGUFs by Unsloth).
   - image generation: Qwen's Qwen-Image-2.1, Microsoft's Mage-Flow, and Qwen3-VL as the text encoder ([IMAGE](docs/IMAGE.md#licenses)).

  Each model's own license applies to its files.
- **Parts it uses:**
   - parts of [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) (MIT); Gyro's formats from agentionai's fork of llama.cpp (MIT);
   - [oneMath](https://github.com/uxlfoundation/oneMath) (Apache-2.0): the rocBLAS backend (the startup comparison and `STRATA_BLAS=1`),
     which CMake fetches from XeStrata's fork with XeStrata's changes ([MistVVK/oneMath](https://github.com/MistVVK/oneMath), under the same license);
   - [intel/llvm](https://github.com/intel/llvm)'s DPC++ (Apache-2.0 WITH LLVM-exception): the compiler and the SYCL runtime,
     built with XeStrata's fixes (`third_party/main/intel-llvm/patches/`, under the same license);
   - [ROCm's device libraries](https://github.com/ROCm/llvm-project/tree/f30ae3e6b680ad23dcf522e668c6d4c5c90838d0/amd/device-libs), by AMD
     (University of Illinois/NCSA): built with intel/llvm, and linked into the code for AMD GPUs; the target features
     their build is given are transcribed from the `debian/rules` of Debian's
     [rocm-llvm](https://salsa.debian.org/rocm-team/rocm-llvm) package (Expat);
   - the image engine's attention: [oneDNN](https://github.com/uxlfoundation/oneDNN) (Apache-2.0, Intel GPUs), [AOTriton](https://github.com/ROCm/aotriton) (MIT, AMD GPUs)
     and [cuDNN Frontend](https://github.com/NVIDIA/cudnn-frontend) (Apache-2.0, parts MIT, NVIDIA GPUs); for reading reference images, [stb_image](https://github.com/nothings/stb) (MIT or the Unlicense);
   - [quantscope](https://github.com/1872183316/quantscope) (MIT or Apache-2.0): what `setup.py --inspect` reads a GGUF's contents with;
   - the web app's font, [Outfit](https://github.com/Outfitio/Outfit-Fonts) (SIL Open Font License 1.1).
- **Ideas from:** those in the original XeStrata's [credits](https://github.com/MistVVK/XeStrata#credits):
   - [Splash](https://github.com/incoai/splash), by Inco AI and the Splash contributors;
   - [ninfer](https://github.com/Neroued/ninfer), by Neroued;
   - [HyperQwen](https://github.com/syv-ai/HyperQwen), by syv.ai and the HyperQwen contributors;
   - [Project Maya](https://github.com/mw00/project-maya), by mw00 and the Project Maya contributors (how many of the
     experts missing from VRAM the CPU computes and how many are read over PCIe);
   - [nextsycl](https://github.com/maxfridbe/nextsycl), by maxfridbe (which kernels the decode of the Qwen3.5 family's
     models merges). The IQ4 lookup's form follows llama.cpp's HIP version.
- The parts are listed in [third_party/main/README.md](third_party/main/README.md); more in [docs/DETAILS.md](docs/DETAILS.md#credits-and-licenses).

## License

XeStrata (copyright MistVVK and the XeStrata contributors) is free software, distributed whole under the
[GNU Lesser General Public License, version 3 or later](COPYING.LESSER) (`COPYING.LESSER` and `COPYING`). So is
this fork.

- **Linking with the makers' compute libraries:** what is linked with a compute library and its runtime libraries
  that a GPU or CPU maker publishes for its own products may be distributed as well: an additional permission under
  section 7 of the GPL, version 3, worded in [NOTICE](NOTICE). ROCm, which this fork uses, is free software, so
  this permission is not needed.
- **Code from Strata and ggml:** the code that comes from Strata and from ggml / llama.cpp, both MIT (the kernels
  transcribed into `src/` included), is under the LGPL as well, as part of XeStrata. Under MIT's conditions the
  copyright notices and the permission notices are kept in [NOTICE](NOTICE).
- **Exceptions:** these, in `third_party/`, stay under their own licenses, one folder per project with its license text.
   - XeStrata's patches to intel/llvm (`third_party/main/intel-llvm/`): Apache-2.0 WITH LLVM-exception.
   - XeStrata's changes to oneMath (its fork, `third_party/main/oneMath/`): Apache-2.0.
   - XeStrata's patch for building AOTriton for Windows (`third_party/main/aotriton/`): AOTriton's license (MIT).
   - Unmodified copies: ggml's `ggml-common.h` and agentionai's llama.cpp's `ggml-common-tq.h` (`third_party/main/ggml/`, MIT),
     `quantscope.py` (`third_party/main/quantscope/`, MIT or Apache-2.0), `stb_image.h` (`third_party/main/stb/`, MIT or the Unlicense).
   - The app's font Outfit (`third_party/main/outfit/`): SIL Open Font License 1.1.
   - `third_party/nonfree/`: what is not free software. The original model's chat template and the experimental
     speed projection's vector, both under the Qwen Community License 1.0.
- **Without `third_party/nonfree/`,** XeStrata still builds and runs. The packages leave it out.
- **What else the packages carry:** intel/llvm's SYCL runtime (Apache-2.0 WITH LLVM-exception), llama.cpp's gguf-py
  (MIT), oneMath (Apache-2.0), oneDNN (Apache-2.0), and AOTriton's library and kernels (MIT; the kernels can carry parts of aiter (MIT), FlyDSL (Apache-2.0) and Triton (MIT)), each with its license text.
  The code for AMD GPUs carries parts of ROCm's device libraries (NCSA), and the image engine stb_image (MIT) and, in the contrib-llvm packages, cuDNN Frontend (Apache-2.0, parts MIT), so their license texts are there too.
  In the contrib-llvm packages the code for NVIDIA GPUs can contain parts of NVIDIA's libdevice, which are under the NVIDIA CUDA Toolkit End User License Agreement.
  Nothing else that is not free software (oneMKL, cuBLAS, cuDNN, NVIDIA's driver) is in them.
- **The models** are not part of this repository; each model's own license applies to its files.
