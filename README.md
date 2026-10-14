<!--
SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# XeStrata-890M

English | [日本語](README.ja.md)

**XeStrata-890M is a fork of [XeStrata](https://github.com/MistVVK/XeStrata) specialised for one GPU: the AMD Radeon 890M (the integrated GPU of the Ryzen AI 9 HX 370, gfx1150, RDNA3.5).**
It runs **[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)**, a 125-billion-parameter AI model,
on a single laptop with a Radeon 890M and 64 GB of RAM.
This GPU is the only target: the other GPUs the original XeStrata runs on are not covered here.

Speed on the 890M (Ryzen AI 9 HX 370, 64 GB RAM, Fedora 44, Qwen3.8-Flash-Next IQ2_XS, APU power 28 W):

| | Original XeStrata | XeStrata-890M |
| --- | ---: | ---: |
| decode (mean of 6 short chats) | 15-22 tok/s | about 31 tok/s |
| prefill, 523 tokens | 5.4 s | 3.2 s |
| prefill, 3,481 tokens | 29.1 s | 16.2 s |
| prefill, 2,669 tokens (one sentence repeated) | 22.0 s | 12.4 s |

The prefill column "Original XeStrata" is the fork at the point where its decode changes were in (the prefill path was still the original's until then).

> **Contents:** [About XeStrata-890M](#about-xestrata-890m) · [What you need](#what-you-need) ·
> [Choosing a model](#choosing-a-model) · [Install](#install) · [Using it](#using-it) · [When something goes wrong](#when-something-goes-wrong) ·
> [How it works](#how-it-works) · [For developers](#for-developers) · [All the details](docs/DETAILS.md)

## About XeStrata-890M

XeStrata is [Strata](https://github.com/Niko1221/Strata) (for NVIDIA GPUs) with its GPU code rewritten in SYCL, an open standard.
On AMD GPUs that SYCL code is compiled with intel/llvm's DPC++ for its HIP target and runs on ROCm.
This fork reworks that AMD path for the Radeon 890M.
The model, the install flow, the pages and the API are Strata's and XeStrata's.

It builds in the free mode only, with free software (ROCm is free software): no oneMKL, no CUDA.
It carries the original XeStrata's changes up to xe0.1.43.1 (Strata 0.1.43).

**What the 890M is, and how the fork uses it:**

- **The CPU and the GPU share one memory (LPDDR5X).** The pool the GPU can take for itself (the GTT) is about half the RAM by the kernel's default.
  The fork puts all 24,576 experts of the model (about 34 GiB) in host memory (host USM) and the GPU reads them there directly.
  No expert is ever computed by the CPU for lack of room on the GPU: decode runs on the GPU alone.
- **Its matrix engine is WMMA (16 x 16 x 16, f16 and int8).** The distribution's hipBLASLt has no kernels for gfx1150 and rocBLAS falls
  back to generic kernels without matrix instructions, so the fork computes the dense matrix products with XeStrata's own WMMA kernel (three times rocBLAS's speed).
- **Power and heat set the speed.** Decode is bounded by the APU's power limit (the PPT, 28 W by default), not by heat.
  33 W makes short chats 3.5% faster, but long generations then reach 92 °C and get slower, so the fork keeps the default 28 W.
  The clock is throttled at 95 °C: compare speeds interleaved, with the GPU cooled down before each run.
- **The CPU, a Ryzen AI 9 HX 370, has AVX-512**, and the CPU-side code takes its AVX-512 paths.

**The main changes for the 890M** (by part, not in commit order):

- The expert cache in host USM (`STRATA_CACHE_HOST=1`, with `--mmap-experts --expert-cache 24576 --no-prefill-borrow`).
  On an integrated GPU the CPU no longer spins while waiting for the GPU, and filling the host USM cache gives the files' pages back to the OS.
- **Decode:**
   - The expert products issue a lane's whole row of loads before decoding (Q2_0 down; IQ2_S, IQ2_XXS and IQ1_M gate/up).
     On AMD the default is one kernel for gate/up and SwiGLU with the activations and the grid tables in local memory (LDS), "mode 8".
     The weights' addresses are built as global-address-space pointers, and the IQ4 table lookup uses `v_perm_b32` with byte selectors.
   - The hyper-connection read-out: the fast version (gr_fast) by default, its BF16 matrices as Q8_0 (`STRATA_HC_Q8`, default on an integrated GPU),
     the up product with 8 lanes per row and 4 rows at once, the down product with 2 rows per sub-group.
   - Sub-group xor exchanges through DPP and `v_permlanex16`.
   - The GDN step and commit stage the tokens' inputs in LDS first and take fewer barriers per token.
   - A Japanese-and-English draft vocabulary (`data/draft_vocab_ja.bin`, 56,554 tokens): the speculative layer reads about half the bytes of the default vocabulary.
   - `STRATA_EXPERT_KEEP=N` (opt-in, changes the answers): only the N heaviest experts of each token are computed.
- **Prefill:**
   - At start the engine times oneMath's library (rocBLAS) against its own WMMA kernel for the dense products and takes the faster
     (HIP GPUs only). `STRATA_BLAS=1` keeps the library, `STRATA_NO_BLAS=1` the own kernel.
   - The own WMMA kernel (mma_gemm) orders its work-groups so that those in flight together cover a compact part of the output
     (blocks of 8 x 8 tiles, `STRATA_MMA_BLOCK`), keeping the rows they read in the 2 MB L2.
   - The experts' int8 products (MMQ) read the blobs in place when every expert a layer routes is in the cache (no gather into group
     buffers, no copies into the MMQ ring; `STRATA_PREFILL_MMQ_INPLACE=0` restores the old way; `STRATA_PREFILL_MMQ_GROUP` is the number
     of experts per launch, default 16). The WMMA body takes no output staging in local memory, so two work-groups fit a WGP.
   - AMD's device compile states each kernel's required work-group size (`STRATA_AMD_REQD_WG`), and the intel/llvm fix
     `hip-reqd-work-group-size` passes it to the backend's register budget (before it every kernel had 128 VGPRs and spilled).
     With that the dense product takes a 64 x 128 tile for a long prompt's large products (`STRATA_MMA_SHAPE=5`: 2048 rows and
     2048 columns or more, and the like), and the MMQ splits its decoders into load and decode and issues a step's loads first,
     keeps the grid tables in local memory and applies the scales with one fma.
   - The expert cache's host USM is, on HIP, memory the GPU may cache (`hipHostMalloc` with NonCoherent; `STRATA_HOST_NC=0` for
     plain host USM): the MMQ reading it in place runs about twice as fast (a 523-token prefill 3.16 -> 2.68 s).
   - The required work-group size is stated for the QSA prompt attention too (PromptAttnMma: 83 spills -> 0, a 3,474-token
     prompt's attention 1.99 -> 1.22 s), for the GDN key-head recurrence (908 -> 602 ms) and for decode's dense GEMV and HC read.
   - The own kernel sums in another order than rocBLAS, and the MMQ's fma rounds its scales differently, so an answer can differ by rounding (`STRATA_BLAS=1` gives the old dense products).

## What you need

| | |
| --- | --- |
| GPU | AMD Radeon 890M (the integrated GPU of the Ryzen AI 9 HX 370, gfx1150). See [AMD GPUs](#amd-gpus). |
| CPU | Ryzen AI 9 HX 370 (12 cores, 24 threads, AVX-512). |
| RAM | 64 GB (IQ2_XS's experts, about 34 GiB, all stay in RAM). See [Choosing a model](#choosing-a-model). |
| Disk | 68 GB for the model and about 6 GB for the MTP layer. An NVMe SSD. |
| OS | Fedora 44 (the distribution's ROCm 7.1.1). |
| BIOS | The memory assigned to the GPU (VRAM) can stay at its default (512 MB): the experts live in host USM, so no larger assignment is needed. The kernel's boot options (`ttm.pages_limit` and the like) stay as they are too. |

### AMD GPUs

The fork targets the Radeon 890M (gfx1150) only.

- Fedora 44's ROCm 7.1.1 rocBLAS has code for gfx1150. Its hipBLASLt has none, which does not matter: the dense products go through the own WMMA kernel
  (the line `rocblaslt error: Cannot read ... TensileLibrary_lazy_gfx1150.dat` in the start-up log is harmless).
- Ubuntu 26.04's rocBLAS has no code for gfx1150: use Fedora 44.
- `./setup.sh --check` prints your GPU's gfx after its name.

## Choosing a model

The fork is run and measured with Qwen3.8-Flash-Next **IQ2_XS** (64 GB of RAM holds all its experts).
To add another size later, run `./setup.sh`.

| Version | Size | Download | RAM | Speed | Quality |
| --- | --- | ---: | ---: | --- | --- |
| Original, Swift 1.5 | **Q2_0** | 66 GB | 48 GB | fastest | good |
| Original, Swift 1.5 | **IQ2_XS** | 68 GB | 48 GB | fast | better (**recommended**) |
| Original, Swift 1.5 | **IQ3_XXS** | 76 GB | 60 GB | slower | very good |
| Original, Swift 1.5 | **IQ3_S** | 84 GB | 62 GB | slow | like the original model |
| Coder | **IQ1_M** | 58 GB | 32 GB | fast | good at code, weak elsewhere |

The RAM figures are the original XeStrata's. The fork's `--expert-cache 24576` (every expert in RAM) takes about 34 GiB with IQ2_XS.
The other sizes have not been run with the fork; where the experts do not all fit in RAM, lower `--expert-cache` (the start-up log shows the free memory).
Unsloth's 4-bit versions (UD-IQ4_XS, UD-Q4_K_XL) have 60-77 GB of experts alone and do not suit a 64 GB PC.

**Versions:**

- **Qwen3.8-Flash-Next**: the original model.
- **[Swift 1.5](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF)**: UkisAI's tune.
  It thinks less before answering, so answers start sooner, at about the same quality. The model's own license applies.
- **[Coder](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF)**: ISTA-DASLab's version for code.
  It keeps the 256 experts used for code of the 512 in each layer. With half the experts it is weaker outside code and outside English.

The other models the original XeStrata runs (Ornith-1.5, llm-jp-4.1, Unsloth's versions) have not been run with the fork
([the original description](https://github.com/MistVVK/XeStrata#choosing-a-model)).

## Install

The fork has no packages: install it from source.

1. On Fedora 44, install the build packages and ROCm (`./setup.sh --packages fedora44/amd --license free` prints the same line):

   ```bash
   sudo dnf install python3 git cmake ninja-build gcc-c++ lld hwloc-devel libzstd-devel libzstd-static \
     vulkan-loader-devel glslc spirv-headers-devel mesa-vulkan-drivers blosc2-devel \
     rocm-hip-devel rocm-device-libs rocm-clang rocm-clang-runtime-devel rocblas-devel hipblaslt-devel hipblas-common-devel
   ```

   Let your user open `/dev/kfd` and `/dev/dri/renderD*` (`sudo usermod -aG render,video $USER`, then log in again).
1. Get the repository and run setup:

   ```bash
   git clone https://github.com/ruc1j/XeStrata-890M.git
   cd XeStrata-890M
   ./setup.sh --license free
   ```

   setup checks the GPU, the RAM and the CPU, asks whether to build intel/llvm here (v7.1.1, with ROCm's HIP target),
   builds the engine, downloads the model and starts it. Enter takes the recommended answer to each question.
   - **Model and size**: Qwen3.8-Flash-Next IQ2_XS.
   - **Context length**: how much text it keeps at once. 32,768 is what the fork is run with.
   - **Images**: whether it should read pictures too.
   - **Experimental speed projection**: an experimental feature that changes how it answers, off by default. Read
     [its description](docs/DETAILS.md#experimental-speed-projection-experimental-off-by-default) before turning it on.
1. After the first start, stop it (close the window) and set the config up for the 890M.
   Open `xestrata-iq2_xs.json`, which setup wrote in the XeStrata folder, set the value after `--expert-cache` in `"args"` to `24576`,
   add `--mmap-experts` and `--no-prefill-borrow`, and add `STRATA_CACHE_HOST` to `"env"`:

   ```json
   "args": [ ..., "--expert-cache", "24576", ..., "--mmap-experts", "--no-prefill-borrow" ],
   "env": { "STRATA_CACHE_HOST": "1" }
   ```

   For Japanese, replace `draft_vocab.bin` in the folder `--mtp` points to with `data/draft_vocab_ja.bin` (the answers stay the same, it is a little faster;
   setup writes the original file back on its next run, so copy the folder under another name, replace the file there and point `--mtp` at it).
1. `./setup.sh` starts it and opens `http://127.0.0.1:8095` in the browser.

- **Time**: the first run downloads 68 GB and builds intel/llvm and the engine, which takes a while. Stopping half-way is fine: the next run continues.
  The driver compiles the GPU code the first time it is used, so the model is run once at the end to get that over with.
- **The PC gets sluggish while the model starts**: it reads about 34 GiB of experts into RAM. The first time takes longest, and
  the PC may respond slowly for 1-3 minutes. Keep the window open and wait.
- **Where things go**: the model files in `XeStrata-data` next to the XeStrata folder; the model's config (`xestrata-<model>.json`) and the logs in the XeStrata folder.
- **RAM headroom**: the engine counts the GPU's free memory as what Linux can give back less 6 GiB (`STRATA_UMA_HEADROOM_GIB`).
  On a 64 GB PC that runs little else, `"STRATA_UMA_HEADROOM_GIB": "3"` in `"env"` leaves more memory for the context.

**From then on** `./setup.sh` starts it right away. Close its window to stop it.
A chat in the terminal is `.venv/bin/python chat.py`.

**Keeping it running while logged in**: `systemctl --user enable --now xestrata` keeps the last configured model running
(no browser is opened). Configure it once with `./setup.sh` first.

**Updating**: `git pull`, then `./update.sh`, which updates the engine, the Python packages and the model's config (it does not start the model).
After an update, or after choosing again in `./setup.sh`, check that the `"args"` and `"env"` you added to `xestrata-iq2_xs.json` are still there.

**Tuning to your PC**: `./setup.sh --calibrate` measures a few engine settings on this PC and keeps the faster ones (5-10 minutes).

**Measuring the speed**: `./setup.sh --bench` measures the installed model's speed on this PC (`--model IQ2_XS` picks the size) and where the time goes,
and writes a record to `records/bench-<time>/` in the data folder (about 40 minutes; stop a running XeStrata first).
What it measures and what the record holds are in [docs/BENCH.md](docs/BENCH.md).

**Removing it**: `./setup.sh --remove-data` deletes the model files, the config and the logs (it shows the sizes and asks); then delete the folder.

The build in detail (the free mode, building intel/llvm, what setup does) is in [docs/BUILD.md](docs/BUILD.md).

## Using it

- **Browser**: `http://127.0.0.1:8095` (opens by itself when the model starts). **Chat**, **Monitor** for the model and the GPU, CPU and RAM,
  **About** for the settings and the addresses.
- **In the terminal**: `.venv/bin/python chat.py`
- **Apps and coding agents**: point an "OpenAI-compatible" client at the base URL **`http://127.0.0.1:8095/v1`**.
  Any API key and model name will do. Apps that speak Anthropic's API use `http://127.0.0.1:8095/v1/messages`.
- **Thinking depth**: the model thinks before it answers. The depth is **Off, Low, Medium or High**, set in the chat page's menu,
  with `/think low` in `chat.py`, or through an app's "reasoning effort" setting. Off is fastest; High suits hard questions.
- **Pictures**: press **Picture** in the chat page; type `/image <path>` in `chat.py`; attach them in an app.
- **From a phone or another PC**: run `./setup.sh --host 0.0.0.0 --api-key <a secret>` and open the address the server window
  shows ([details](docs/DETAILS.md#streaming-and-connecting)).

**Good to know**: requests are handled one at a time (`"parallel": 2` handles several at once, [BATCHING](docs/BATCHING.md)). The first message of a
conversation is read in full; after that the conversation is remembered and only what was added is read, so the next reply starts right away.

## When something goes wrong

**The PC froze or got very slow at the first start**
Normal while starting, especially the first time: the experts are being read into RAM.
Keep the window open and wait. If it is still frozen after 10 minutes, restart the PC, close other programs
(the browser above all) and try again.

**The download or the install stopped half-way**
Run `./setup.sh` again. It continues where it stopped.

**It says the GPU was not found or cannot be used**
Check these two things.

- ROCm's libraries (HIP, rocBLAS, hipBLASLt) are installed: the package line under [Install](#install).
- Your user can open `/dev/kfd` and `/dev/dri/renderD*` (the `render` and `video` groups).

**The start-up log says `rocblaslt error: Cannot read ... TensileLibrary_lazy_gfx1150.dat`**
Harmless. hipBLASLt has no kernels for gfx1150; the dense products go through the own WMMA kernel.

**It says port 8095 is already in use**
XeStrata is already running: find its window. If another program holds the port, pick another with `./setup.sh --port 8081`.

**It is very slow and the disk light stays on**
Not enough RAM. Close other programs, lower `--expert-cache`, or choose the smaller size (Q2_0).

**"the engine stopped unexpectedly" appears in the middle of an answer**
Usually out of RAM (Linux stopped the engine). Send the same message again: the engine restarts by itself.
If it keeps happening, close other programs or lower `--expert-cache`.

**It slows down during a long generation**
The GPU is throttled when it reaches 95 °C. Keep the laptop's air intake clear.
Raising the APU's power limit (the PPT) does not help long generations: they get slower from the heat (28 W is recommended).

**It says the prompt exceeds the context length**
The conversation is longer than the context you chose. Start a new conversation, or choose a longer context in `./setup.sh`.

**Still stuck**
See the [detailed table](docs/DETAILS.md#troubleshooting). If that does not solve it, open an
[issue](https://github.com/ruc1j/XeStrata-890M/issues) and attach `xestrata-<model>.log` from the XeStrata folder.

## How it works

A model like this normally runs on a server with hundreds of GB of GPU memory. On the 890M the CPU and the GPU share one RAM,
so XeStrata-890M **uses the whole RAM as the GPU's memory**.

- **The model is a collection of 24,576 small specialists (experts)**, and writing one word uses just 10 of them.
  The fork keeps every expert (about 34 GiB with IQ2_XS) in host memory, where the GPU reads them directly.
  The GPU's pool (the GTT) holds the layers every word needs, the KV cache and the working memory.
- **The GPU** does all the work: the computation every word needs and the experts'. In decode the CPU does little.
- **The SSD** holds a large table, of which a few rows are read per word.
- **Guess, then check**: a small helper built into the model (the MTP layer) guesses the next few words, and the big model checks them in one go.
  The right ones are kept, and it writes the next word itself. The big model always decides, so the answers' quality is unchanged.
- **Long texts are read in big chunks** (up to 32,768 tokens at once). The dense products of prefill run on the 890M's WMMA,
  the expert products on its int8 WMMA.

Each part is described in [docs/DETAILS.md](docs/DETAILS.md#how-it-works) (the original XeStrata's description, which also covers other GPUs).

## For developers

- **Build**: the free mode only (`build/free`, `-DSTRATA_LICENSE=free`, intel/llvm v7.1.1 built with HIP by `tools/intel_llvm_build.py`).
  The steps are in [docs/BUILD.md](docs/BUILD.md). This machine's facts (paths, settings, tools) go into `LOCAL.md`, which git ignores.
- **Kernel benchmarks**: `build/free/gemm_bench` (the dense products), `iq_mmq_parity --bench` (the experts' int8 products), `expert_bench`, `gdn_bench`, `gr_bench`, `mmvq_bench`.
  Measure kernels at the high clock (`dpm high`), not a fixed one; A/B runs interleaved under the same conditions, with the GPU cooled down first (heat moves them by a few per cent).
  `STRATA_PREFILL_TIMING=1` logs the GPU time of each prefill stage.
- **Reading the ISA**: emit bitcode with `-fsycl-device-only`, turn it into assembly with intel/llvm's `llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1150`,
  and read `.vgpr_count` and `.vgpr_spill_count` in the metadata. This intel/llvm's AMD backend caps every kernel at 128 VGPRs, so large tiles spill.
- **Rules**: [AGENTS.md](AGENTS.md). A branch for the 890M is chosen from what the device reports (an integrated GPU, the HIP backend, WMMA) or an opt-in environment variable, not from the GPU's name.
  The lints (`tools/lint/run.sh`), the tests (CTest, `python -m unittest`) and the license notices are there too.
  Measurement records do not go into git.
- **README and docs**: the Japanese files (README.ja.md, docs/*.ja.md) are written first; the English ones (README.md, docs/*.md) are their translations.
  `docs/` is the original XeStrata's and covers other GPUs as well.

## Credits

- **The original software**: [XeStrata](https://github.com/MistVVK/XeStrata) by MistVVK and the XeStrata contributors,
  and its original, [Strata](https://github.com/Niko1221/Strata) by Niko1221 and the Strata contributors.
- **Models**:
   - [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) by the Qwen team.
   - The compressed versions by [ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF), and their
     [Coder](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF).
   - [Swift 1.5](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF) by UkisAI.

  The model files are under their own licenses.
- **Components**:
   - Parts of [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) (MIT).
   - [oneMath](https://github.com/uxlfoundation/oneMath) (Apache-2.0): the rocBLAS backend (the start-up comparison and `STRATA_BLAS=1`).
     CMake fetches the fork with XeStrata's changes ([MistVVK/oneMath](https://github.com/MistVVK/oneMath), same license).
   - [intel/llvm](https://github.com/intel/llvm)'s DPC++ (Apache-2.0 WITH LLVM-exception): the compiler and the SYCL runtime,
     built with XeStrata's patches (`third_party/main/intel-llvm/patches/`, same license).
   - The pages' font [Outfit](https://github.com/Outfitio/Outfit-Fonts) (SIL Open Font License 1.1).
- **Ideas**: those in the original XeStrata's [credits](https://github.com/MistVVK/XeStrata#credits).
  The IQ4 table lookup follows llama.cpp's HIP version.
- More in [docs/DETAILS.md](docs/DETAILS.md#credits-and-licenses).

## License

XeStrata (copyright MistVVK and the XeStrata contributors) is free software, distributed as a whole under the
[GNU Lesser General Public License, version 3 or later](COPYING.LESSER) (`COPYING.LESSER` and `COPYING`). So is this fork.

- **Linking with a maker's compute library**: XeStrata linked with a GPU or CPU maker's compute library for its own products and its runtime libraries may be conveyed too.
  This is an additional permission under section 7 of GPL version 3; the text is in [NOTICE](NOTICE). The fork's ROCm is free software, so it does not need this permission.
- **Code from Strata and ggml**: the code that came from the MIT-licensed Strata and ggml / llama.cpp (including the kernels transcribed into `src/`)
  is part of XeStrata under the LGPL. As the MIT terms require, their copyright and permission notices are kept in [NOTICE](NOTICE).
- **Exceptions**: these under `third_party/` keep their own licenses, each in its project's folder with the license text.
   - XeStrata's patches to intel/llvm (`third_party/main/intel-llvm/`): Apache-2.0 WITH LLVM-exception.
   - XeStrata's changes to oneMath (the fork, `third_party/main/oneMath/`): Apache-2.0.
   - ggml's `ggml-common.h` (`third_party/main/ggml/`, an unchanged copy): MIT.
   - The pages' font Outfit (`third_party/main/outfit/`): SIL Open Font License 1.1.
   - `third_party/nonfree/`: what is not free software. The original model's chat template and the experimental speed projection's vectors,
     both under the Qwen Community License 1.0.
- **Without `third_party/nonfree/`** XeStrata still builds and runs.
- **The model**: not in this repository. The model files are under their own licenses.
