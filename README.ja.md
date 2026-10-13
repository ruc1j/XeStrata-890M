<!--
SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# XeStrata-890M

[English](README.md) | 日本語

**XeStrata-890M は、[XeStrata](https://github.com/MistVVK/XeStrata) を AMD Radeon 890M（Ryzen AI 9 HX 370 の内蔵 GPU、gfx1150、RDNA3.5）だけに特化させたフォークです。**
1,250 億パラメーターの AI モデル **[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)** を、
Radeon 890M と 64 GB の RAM を積んだノート PC 1 台で動かします。
対象はこの GPU だけです。元の XeStrata が扱うほかの GPU のことは、このフォークでは扱いません。

890M での速さ（Ryzen AI 9 HX 370、RAM 64 GB、Fedora 44、Qwen3.8-Flash-Next IQ2_XS、APU の電力 28 W）:

| | 元の XeStrata | XeStrata-890M |
| --- | ---: | ---: |
| decode（短い会話 6 つの平均） | 15〜22 tok/s | 約 31 tok/s |
| prefill 523 トークン | 5.4 s | 3.2 s |
| prefill 3,481 トークン | 29.1 s | 16.2 s |
| prefill 2,669 トークン（同じ文の繰り返し） | 22.0 s | 12.4 s |

prefill の「元の XeStrata」の列は、このフォークの decode の改修を入れた時点の値です（prefill の経路はそこまで元のままでした）。

> **目次:** [XeStrata-890M とは](#xestrata-890m-とは) · [必要なもの](#必要なもの) · [モデルの選び方](#モデルの選び方) ·
> [インストール](#インストール) · [使い方](#使い方) · [困ったとき](#困ったとき) · [しくみ](#しくみ) ·
> [開発する人へ](#開発する人へ) · [詳しい説明](docs/DETAILS.ja.md)

## XeStrata-890M とは

XeStrata は、[Strata](https://github.com/Niko1221/Strata)（NVIDIA の GPU 向け）の GPU の計算を、オープンな標準の SYCL に書き直したものです。
AMD の GPU では、intel/llvm の DPC++ の HIP のターゲットでその SYCL のコードをコンパイルし、ROCm の上で動かします。
このフォークは、その AMD の経路を Radeon 890M に合わせて作り直したものです。
モデル、インストールの流れ、画面、API は Strata と XeStrata のままです。

ビルドは自由ソフトウェアだけの free 方式だけです（ROCm は自由ソフトウェアです）。oneMKL や CUDA は使いません。
元の XeStrata の xe0.1.43.1（Strata 0.1.43）までの変更を取り込んでいます。

**890M の性質と、このフォークの使い方:**

- **CPU と GPU が同じメモリ（LPDDR5X）を使います。** GPU が自分で確保できる共有のプール（GTT）は、カーネルの既定では RAM の約半分です。
  このフォークは、モデルの 24,576 個のエキスパート（約 34 GiB）をすべてホストのメモリ（host USM）に置き、GPU がそこを直接読みます。
  GPU にないエキスパートを CPU が計算することはなくなり、decode は GPU だけで回ります。
- **行列エンジンは WMMA（16 x 16 x 16 の f16 と int8）です。** ディストリビューションの hipBLASLt は gfx1150 のカーネルを持たず、
  rocBLAS は行列命令を使わない汎用カーネルに落ちます。このフォークは密な行列積を XeStrata 自身の WMMA のカーネルで計算します（rocBLAS の 3 倍の速さ）。
- **電力と熱で速さが決まります。** decode の速さは APU の電力の上限（PPT、既定で 28 W）で決まり、熱ではありません。
  33 W にすると短い会話は 3.5% 速くなりますが、長い生成では 92 度まで上がって遅くなるので、既定の 28 W のままにしています。
  95 度でクロックが絞られるので、速さの比較は交互に、GPU を冷ましてから測ります。
- **CPU の Ryzen AI 9 HX 370 は AVX-512 を持ち**、CPU 側の計算は AVX-512 の経路を使います。

**890M 向けの主な変更**（コミットの順ではなく、部分ごと）:

- エキスパートのキャッシュを host USM に（`STRATA_CACHE_HOST=1`、`--mmap-experts --expert-cache 24576 --no-prefill-borrow` と組み合わせます）。
  統合 GPU では、GPU を待つ間に CPU を空回りさせず、host USM のキャッシュを詰めるときにファイルのページを OS に返します。
- **decode:**
   - エキスパートの積: 1 レーンが受け持つ行のロードを先にまとめて出します（Q2_0 の down、IQ2_S・IQ2_XXS・IQ1_M の gate/up）。
     AMD では gate/up と SwiGLU を 1 つのカーネルで計算し、活性と格子表をローカルメモリ（LDS）に置く形（mode 8）を既定にしています。
     重みのアドレスはグローバルアドレス空間のポインタとして作り、IQ4 の表引きは `v_perm_b32` のバイト指定で行います。
   - ハイパーコネクションの読み出し: 高速版（gr_fast）を既定にし、BF16 の行列を Q8_0 にして（`STRATA_HC_Q8`、統合 GPU で既定）、
     up は 1 行 8 レーンで 4 行を同時に、down は 1 サブグループ 2 行で計算します。
   - サブグループの xor の交換を DPP と `v_permlanex16` で行います。
   - GDN の step と commit は、トークンの入力を先に LDS へ載せ、トークンごとのバリアを減らします。
   - 日本語と英語の draft の語彙（`data/draft_vocab_ja.bin`、56,554 トークン）: 推測の層が読むバイトが既定の語彙の約半分になります。
   - `STRATA_EXPERT_KEEP=N`（opt-in、答えが変わります）: トークンごとに重い順の N 個のエキスパートだけを計算します。
- **prefill:**
   - 密な行列積は、起動時に oneMath のライブラリ（rocBLAS）と自身の WMMA のカーネルの速さを比べて速い方を取ります
     （HIP の GPU のみ）。`STRATA_BLAS=1` でライブラリ、`STRATA_NO_BLAS=1` で自身のカーネルに固定できます。
   - 自身の WMMA のカーネル（mma_gemm）は、同時に走るワークグループが出力の小さな四角を受け持つ順（8 x 8 タイルのブロック、`STRATA_MMA_BLOCK`）で、
     2 MB の L2 に読む行が残るようにします。
   - エキスパートの int8 の積（MMQ）は、層のエキスパートがすべてキャッシュにあれば blob をその場で読みます（グループへの gather や
     MMQ ring へのコピーを省く、`STRATA_PREFILL_MMQ_INPLACE=0` で元に戻せます。1 回に積むエキスパートの数は `STRATA_PREFILL_MMQ_GROUP`、既定 16）。
     WMMA 版は出力の staging をローカルメモリに取らず、WGP に 2 ワークグループが載るようにします。
   - 自身のカーネルは rocBLAS と足す順が違うので、答えが丸めの差で変わることがあります（`STRATA_BLAS=1` なら元のまま）。

## 必要なもの

| | |
| --- | --- |
| GPU | AMD Radeon 890M（Ryzen AI 9 HX 370 の内蔵 GPU、gfx1150）。[AMD の GPU](#amd-の-gpu) も見てください。 |
| CPU | Ryzen AI 9 HX 370（12 コア 24 スレッド、AVX-512）。 |
| RAM | 64 GB（IQ2_XS のエキスパート約 34 GiB をすべて RAM に置きます）。[モデルの選び方](#モデルの選び方)。 |
| ディスク | モデルに 68 GB、MTP 層に約 6 GB。NVMe の SSD。 |
| OS | Fedora 44（ディストリビューションの ROCm 7.1.1）。 |
| BIOS | GPU へのメモリの割り当て（VRAM）は既定（512 MB）のままで構いません。エキスパートは host USM に置くので、割り当てを増やす必要はありません。カーネルの起動オプション（`ttm.pages_limit` など）も変えません。 |

### AMD の GPU

このフォークが対象にするのは Radeon 890M（gfx1150）だけです。

- Fedora 44 の ROCm 7.1.1 の rocBLAS は gfx1150 のコードを持ちます。hipBLASLt は持ちませんが、密な行列積は自身の WMMA のカーネルで計算するので構いません
  （起動のログの `rocblaslt error: Cannot read ... TensileLibrary_lazy_gfx1150.dat` は無害です）。
- Ubuntu 26.04 の rocBLAS は gfx1150 のコードを持ちません。Fedora 44 を使ってください。
- 自分の GPU の gfx は、`./setup.sh --check` が GPU の名前の後ろに出します。

## モデルの選び方

このフォークで動かして測っているのは、Qwen3.8-Flash-Next の **IQ2_XS** です（64 GB の RAM に、エキスパートをすべて置けます）。
あとから別の大きさを足すときは `./setup.sh` を実行します。

| 版 | 大きさ | ダウンロード | RAM の目安 | 速さ | 品質 |
| --- | --- | ---: | ---: | --- | --- |
| 元のモデル、Swift 1.5 | **Q2_0** | 66 GB | 48 GB | いちばん速い | 良い |
| 元のモデル、Swift 1.5 | **IQ2_XS** | 68 GB | 48 GB | 速い | より良い（**おすすめ**） |
| 元のモデル、Swift 1.5 | **IQ3_XXS** | 76 GB | 60 GB | 遅め | とても良い |
| 元のモデル、Swift 1.5 | **IQ3_S** | 84 GB | 62 GB | 遅い | 元のモデルと同等 |
| Coder | **IQ1_M** | 58 GB | 32 GB | 速い | コードでは良い、ほかは弱い |

RAM の目安は元の XeStrata のものです。このフォークの `--expert-cache 24576`（エキスパートをすべて RAM に置く）は、IQ2_XS で約 34 GiB を使います。
ほかの大きさはこのフォークでは動かしていません。エキスパートの全部を RAM に置けなければ、`--expert-cache` を小さくします（起動時のログの空きのメモリを見て決めます）。
Unsloth の 4 ビットの版（UD-IQ4_XS、UD-Q4_K_XL）は、エキスパートだけで 60〜77 GB あり、64 GB の PC には向きません。

**版:**

- **Qwen3.8-Flash-Next**: 元のモデル。
- **[Swift 1.5](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF)**: UkisAI による調整版です。
  答える前の思考が短く、答えが早く出ます。品質はほぼ同じです。このモデル自身のライセンスに従います。
- **[Coder](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF)**: ISTA-DASLab によるコード向けの版です。
  各層 512 個のエキスパートのうち、コードで使う 256 個だけを残しています。エキスパートが半分なので、コード以外や、英語以外の言語では弱くなります。

元の XeStrata が扱うほかのモデル（Ornith-1.5、llm-jp-4.1、Unsloth の版）は、このフォークでは動かしていません
（[元の説明](https://github.com/MistVVK/XeStrata#choosing-a-model)）。

## インストール

このフォークにはパッケージがありません。ソースから入れます。

1. Fedora 44 で、ビルドに要るパッケージと ROCm を入れます（`./setup.sh --packages fedora44/amd --license free` も同じ 1 行を出します）:

   ```bash
   sudo dnf install python3 git cmake ninja-build gcc-c++ lld hwloc-devel libzstd-devel libzstd-static \
     vulkan-loader-devel glslc spirv-headers-devel mesa-vulkan-drivers blosc2-devel \
     rocm-hip-devel rocm-device-libs rocm-clang rocm-clang-runtime-devel rocblas-devel hipblaslt-devel hipblas-common-devel
   ```

   自分のユーザーが `/dev/kfd` と `/dev/dri/renderD*` を開けるようにします（`sudo usermod -aG render,video $USER` のあと、ログインし直します）。
1. リポジトリを取ってきて、setup を実行します:

   ```bash
   git clone https://github.com/ruc1j/XeStrata-890M.git
   cd XeStrata-890M
   ./setup.sh --license free
   ```

   setup は GPU と RAM と CPU を確かめ、intel/llvm（v7.1.1、ROCm の HIP のターゲット付き）をここでビルドするかを尋ね、
   エンジンをビルドし、モデルをダウンロードして起動します。質問は Enter を押せばおすすめの選択になります。
   - **モデルと大きさ**: Qwen3.8-Flash-Next の IQ2_XS。
   - **文脈の長さ**: 一度に覚えておける文章の量です。32,768 を使っています。
   - **画像**: 画像も読ませるかどうか。
   - **実験的な速度向上用の射影**: 答え方が変わる実験的な機能で、既定ではオフです。使う前に
     [説明](docs/DETAILS.ja.md#実験的な速度向上用の射影実験的既定はオフ)を読んでください。
1. 一度起動したら止め（ウィンドウを閉じます）、設定を 890M 向けにします。
   setup が XeStrata のフォルダーに書いた `xestrata-iq2_xs.json` を開き、`"args"` の `--expert-cache` の値を `24576` にして
   `--mmap-experts` と `--no-prefill-borrow` を足し、`"env"` に `STRATA_CACHE_HOST` を足します:

   ```json
   "args": [ ..., "--expert-cache", "24576", ..., "--mmap-experts", "--no-prefill-borrow" ],
   "env": { "STRATA_CACHE_HOST": "1" }
   ```

   日本語で使うなら、`--mtp` の指すフォルダーの `draft_vocab.bin` を `data/draft_vocab_ja.bin` で置き換えます（答えは変わらず、少し速くなります。
   元のファイルは setup が次に書き戻すので、別の名前のフォルダーに写してから置き換え、`--mtp` をそこに向けます）。
1. `./setup.sh` を実行すると起動し、ブラウザで `http://127.0.0.1:8095` が開きます。

- **時間**: 初回はダウンロード（68 GB）と intel/llvm とエンジンのビルドに時間がかかります。途中で止めても、次は続きから始まります。
  GPU のコードは初めて使うときにドライバーがコンパイルするので、最後にモデルを一度動かしてそれを済ませます。
- **起動中は PC が重くなります**: モデルを起動すると、エキスパート約 34 GiB を RAM に読み込みます。初回はとくに時間がかかり、
  1〜3 分ほど PC の反応が遅くなることがあります。ウィンドウを閉じずに待ってください。
- **置き場所**: モデルのファイルは XeStrata のフォルダーの隣の `XeStrata-data`、モデルの設定（`xestrata-<モデル>.json`）とログは XeStrata のフォルダーに置きます。
- **RAM の余裕**: エンジンは、Linux が返せるメモリから 6 GiB（`STRATA_UMA_HEADROOM_GIB`）を引いた分を GPU の空きとして数えます。
  64 GB の PC でほかのプログラムをあまり動かさないなら、`"env"` に `"STRATA_UMA_HEADROOM_GIB": "3"` を足すと文脈に使えるメモリが増えます。

**次からは** `./setup.sh` を実行すると、すぐに起動します。止めるときはそのウィンドウを閉じます。
端末での会話は `.venv/bin/python chat.py` です。

**ログインしている間ずっと動かす**: `systemctl --user enable --now xestrata` で、最後に設定したモデルを起動しておきます
（ブラウザは開きません）。先に `./setup.sh` で一度設定しておきます。

**更新**: `git pull` のあと `./update.sh` を実行すると、エンジンと Python のパッケージとモデルの設定を更新します（モデルは起動しません）。
更新や `./setup.sh` での選び直しのあとは、`xestrata-iq2_xs.json` に足した `"args"` と `"env"` が残っているか確かめます。

**自分の PC に合わせる**: `./setup.sh --calibrate` は、エンジンの設定のいくつかをこの PC で測り、速いものを残します（5〜10 分ほど）。

**速さを測る**: `./setup.sh --bench` は、入れたモデル（`--model IQ2_XS` などで指定）のこの PC での速さと、どこに時間がかかっているかを測り、
データフォルダの `records/bench-<日時>/` に記録を書きます（40 分ほど。動いている XeStrata は先に止めます）。
測るものと記録の中身は [docs/BENCH.ja.md](docs/BENCH.ja.md) にあります。

**消す**: `./setup.sh --remove-data` でモデルのファイル・設定・ログを消してから（大きさを見せて確かめます）、フォルダーを消します。

ビルドの詳しい説明（free 方式、intel/llvm のビルド、setup の振る舞い）は [docs/BUILD.ja.md](docs/BUILD.ja.md) にあります。

## 使い方

- **ブラウザ**: `http://127.0.0.1:8095`（モデルが起動すると自動で開きます）。**Chat**、モデルと GPU・CPU・RAM の様子を
  見る **Monitor**、設定とアドレスを見る **About** があります。
- **端末で会話**: `.venv/bin/python chat.py`
- **アプリやコーディングエージェント**: 「OpenAI 互換」の接続先として、ベース URL **`http://127.0.0.1:8095/v1`** を指定します。
  API キーとモデル名は何でも構いません。Anthropic の API を使うアプリは `http://127.0.0.1:8095/v1/messages` です。
- **思考の深さ**: モデルは答える前に考えます。深さは **Off、Low、Medium、High** から選べます。チャットの画面のメニュー、
  `chat.py` の `/think low`、アプリの「reasoning effort」の設定で変えられます。Off がいちばん速く、難しい質問には High が向きます。
- **画像**: チャットの画面では **Picture** を押します。`chat.py` では `/image <パス>` と入力します。アプリでは添付します。
- **スマートフォンや別の PC から**: `./setup.sh --host 0.0.0.0 --api-key <秘密の文字列>` を実行し、
  サーバーのウィンドウに出るアドレスを開きます（[詳細](docs/DETAILS.ja.md#ストリーミングと接続)）。

**知っておくと良いこと**: 要求は 1 つずつ処理します（複数を同時に処理するには `"parallel": 2`、[BATCHING](docs/BATCHING.ja.md)）。会話の最初のメッセージはすべて読み込みますが、
そのあとは会話を覚えておき、新しく足された部分だけを読むので、続きの返事はすぐに始まります。

## 困ったとき

**最初の起動で PC が固まった、とても遅くなった**
起動中、とくに初回はよくあります。エキスパートを RAM に読み込んでいるところです。
ウィンドウを閉じずに待ってください。10 分たっても固まったままなら、PC を再起動し、ほかのプログラム
（とくにブラウザ）を閉じてからやり直してください。

**ダウンロードやインストールの途中で止まった**
`./setup.sh` をもう一度実行してください。止まったところから続けます。

**GPU が見つからない、使えないと言われた**
次の 2 つを確かめてください。

- ROCm のライブラリ（HIP、rocBLAS、hipBLASLt）が入っているか。[インストール](#インストール)のパッケージの行です。
- 自分のユーザーで `/dev/kfd` と `/dev/dri/renderD*` を開けるか（`render` と `video` のグループ）。

**起動のログに `rocblaslt error: Cannot read ... TensileLibrary_lazy_gfx1150.dat` と出る**
無害です。hipBLASLt が gfx1150 のカーネルを持たないという知らせで、密な行列積は自身の WMMA のカーネルで計算します。

**ポート 8095 はすでに使われていると言われた**
XeStrata がすでに動いています。そのウィンドウを探してください。ほかのプログラムが使っているなら、
`./setup.sh --port 8081` のように別のポートを選べます。

**とても遅く、ディスクのランプが点きっぱなし**
RAM が足りていません。ほかのプログラムを閉じるか、`--expert-cache` を小さくするか、小さい大きさ（Q2_0）を選んでください。

**答えの途中で「the engine stopped unexpectedly」と出た**
たいていは RAM 不足です（Linux がエンジンを止めます）。同じメッセージをもう一度送れば、エンジンは自動で起動し直します。
何度も起きるなら、ほかのプログラムを閉じるか、`--expert-cache` を小さくしてください。

**長い生成の途中で遅くなる**
GPU が 95 度に達するとクロックが絞られます。ノート PC の吸気を塞がないでください。
APU の電力の上限（PPT）を上げても、長い生成では熱で遅くなります（28 W のままを勧めます）。

**プロンプトが文脈の長さを超えると言われた**
会話が、選んだ文脈の長さより長くなっています。新しい会話を始めるか、`./setup.sh` で長い文脈を選び直してください。

**解決しないとき**
[詳しい対処の表](docs/DETAILS.ja.md#困ったとき)を見てください。それでも解決しなければ
[issue](https://github.com/ruc1j/XeStrata-890M/issues) を立て、XeStrata のフォルダーの `xestrata-<モデル>.log` を添付してください。

## しくみ

このようなモデルは、ふつう数百 GB の GPU のメモリを持つサーバーで動かします。890M では CPU と GPU が 1 つの RAM を分け合うので、
XeStrata-890M は **RAM 全体を GPU のメモリとして使います**。

- **モデルは 24,576 個の小さな専門家（エキスパート）の集まり**で、1 語を書くのに使うのはそのうち 10 個だけです。
  このフォークは、すべてのエキスパート（IQ2_XS で約 34 GiB）をホストのメモリに置き、GPU が直接読みます。
  GPU のプール（GTT）には、どの語にも必要な層、KV キャッシュ、作業用のメモリを置きます。
- **GPU** が、どの語にも必要な計算とエキスパートの計算をすべて受け持ちます。decode では CPU はほとんど働きません。
- **SSD** には大きな表を置き、1 語ごとにその数行だけを読みます。
- **推測してから確かめる**: モデルに組み込まれた小さな助手（MTP 層）が次の数語を推測し、大きなモデルがそれをまとめて確かめます。
  合っているものは残し、次の 1 語は自分で書きます。決めるのはいつも大きなモデルなので、答えの品質は変わりません。
- **長い文章は大きな塊で読みます**（一度に最大 32,768 トークン）。prefill の密な行列積は 890M の WMMA で、
  エキスパートの積は int8 の WMMA で計算します。

各部分の詳しい説明は [docs/DETAILS.ja.md](docs/DETAILS.ja.md#しくみ) にあります（元の XeStrata の説明で、ほかの GPU の記述も含みます）。

## 開発する人へ

- **ビルド**: free 方式だけです（`build/free`、`-DSTRATA_LICENSE=free`、intel/llvm v7.1.1 を `tools/intel_llvm_build.py` で HIP 付きにビルドしたもの）。
  手順は [docs/BUILD.ja.md](docs/BUILD.ja.md) にあります。この機械の事実（パス、設定、道具）は `LOCAL.md`（git には入れません）に書きます。
- **カーネル単体の計測**: `build/free/gemm_bench`（密な積）、`iq_mmq_parity --bench`（エキスパートの int8 の積）、`expert_bench`、`gdn_bench`、`gr_bench`、`mmvq_bench`。
  単体はクロックを固定せず高い側（`dpm high`）で測り、A/B は同じ条件で交互に、GPU を冷ましてから測ります（熱で数 % 動きます）。
  `STRATA_PREFILL_TIMING=1` で prefill の段階ごとの GPU 時間がログに出ます。
- **ISA の確認**: `-fsycl-device-only` で bitcode を出し、intel/llvm の `llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1150` でアセンブリにして、
  メタデータの `.vgpr_count` と `.vgpr_spill_count` を見ます。この intel/llvm の AMD の backend はすべてのカーネルを 128 VGPR に抑えるので、
  大きなタイルは溢れます。
- **守ること**: [AGENTS.md](AGENTS.md)。890M 向けの分岐は、GPU の名前ではなく、報告される性質（統合 GPU、HIP の backend、WMMA の有無）か環境変数の opt-in で選びます。
  リント（`tools/lint/run.sh`）、テスト（CTest、`python -m unittest`）、ライセンスの表示もそこにあります。
  計測の記録は git に入れません。
- **README と docs**: 日本語版（README.ja.md、docs/*.ja.md）を先に書き、英語版（README.md、docs/*.md）はその訳です。
  `docs/` は元の XeStrata のもので、ほかの GPU の記述を含みます。

## クレジット

- **元のソフトウェア**: MistVVK と XeStrata の貢献者による [XeStrata](https://github.com/MistVVK/XeStrata)、
  その元の、Niko1221 と Strata の貢献者による [Strata](https://github.com/Niko1221/Strata)。
- **モデル**:
   - Qwen チームの [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)。
   - [ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) による圧縮版と、
     [Coder](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF)。
   - UkisAI による [Swift 1.5](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF)。

  モデルのファイルには、それぞれのライセンスが適用されます。
- **使っている部品**:
   - [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp)（MIT）の一部。
   - [oneMath](https://github.com/uxlfoundation/oneMath)（Apache-2.0）: rocBLAS の後端（起動時の比較と `STRATA_BLAS=1`）。
     CMake が XeStrata の変更の入ったフォーク（[MistVVK/oneMath](https://github.com/MistVVK/oneMath)、同じライセンス）から取得します。
   - [intel/llvm](https://github.com/intel/llvm) の DPC++（Apache-2.0 WITH LLVM-exception）: コンパイラと SYCL の実行時。
     XeStrata の修正（`third_party/main/intel-llvm/patches/`、同じライセンス）を当ててビルドします。
   - 画面のフォント [Outfit](https://github.com/Outfitio/Outfit-Fonts)（SIL Open Font License 1.1）。
- **参考にした考え方**: 元の XeStrata の[クレジット](https://github.com/MistVVK/XeStrata#credits)にあるもの。
  IQ4 の表引きの形は llama.cpp の HIP 版を参考にしています。
- 詳しくは [docs/DETAILS.ja.md](docs/DETAILS.ja.md#クレジットとライセンス) にあります。

## ライセンス

XeStrata（著作権は MistVVK と XeStrata の貢献者）は自由ソフトウェアで、全体を
[GNU Lesser General Public License の第 3 版以降](COPYING.LESSER)（`COPYING.LESSER` と `COPYING`）で配布します。このフォークも同じです。

- **メーカーの計算ライブラリとのリンク**: GPU や CPU のメーカーが自社の製品向けに出している計算ライブラリとその実行時ライブラリとリンクしたものも配布できます。
  GPL 第 3 版の第 7 条による追加の許可で、文面は [NOTICE](NOTICE) にあります。このフォークが使う ROCm は自由ソフトウェアなので、この許可は要りません。
- **Strata と ggml から来たコード**: MIT ライセンスの Strata と ggml / llama.cpp から来たコード（`src/` に書き写したカーネルを含む）も、
  XeStrata の一部として LGPL です。MIT の条件どおり、著作権表示と許諾表示を [NOTICE](NOTICE) に残しています。
- **例外**: `third_party/` の次のものは、元のライセンスのままです。プロジェクトごとのフォルダーに、ライセンスの文書と一緒に置いています。
   - intel/llvm への XeStrata の修正のパッチ（`third_party/main/intel-llvm/`）: Apache-2.0 WITH LLVM-exception。
   - oneMath への XeStrata の変更（フォーク、`third_party/main/oneMath/`）: Apache-2.0。
   - ggml の `ggml-common.h`（`third_party/main/ggml/`、変えていない写し）: MIT。
   - 画面のフォント Outfit（`third_party/main/outfit/`）: SIL Open Font License 1.1。
   - `third_party/nonfree/`: 自由ソフトウェアでないもの。元のモデルのチャットテンプレートと、実験的な速度向上用の射影のベクトルで、
     どちらも Qwen Community License 1.0 です。
- **`third_party/nonfree/` がなくても**、XeStrata はビルドでき、動きます。
- **モデル**: このリポジトリには含みません。モデルのファイルには、それぞれのライセンスが適用されます。
