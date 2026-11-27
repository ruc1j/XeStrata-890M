<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# Intel の GPU での XeStrata

[English](XE.md) | 日本語

この文書は XeStrata のエンジンを Intel の GPU で動かすための実装と、それをどう確かめたかの記録です。
エンジンが GPU に求めること、移植した計算と検証の一覧をまとめます。
使い方は [README](../README.ja.md) と [詳しい説明](DETAILS.ja.md)、ビルドの方法、setup の振る舞い、要るパッケージは [BUILD](BUILD.ja.md) にあります。

XeStrata xe0.2.22-1 は、Strata のエンジンを Level Zero と SYCL で Intel の GPU に移したものです。版は、取り込んだ upstream の版と、その版での XeStrata のリリースの回数（1 から）を「-」でつないだものです（xe0.2.8-1。upstream のベータ版のような先行の版は、deb と rpm のパッケージの版で「-」を「~」にします: xe0.2.0-beta-1 は 0.2.0~beta-1 で、正式版の 0.2.0-1 より前の版として扱われ、そのまま上げられます。2026-10-13 より前のリリースは、2 回目から末尾に番号を足していました: xe0.1.40.2.1）。
移植の元は Strata 0.1.24（`3ce2523c2823687de5372be3af58534f56cbf286`）で、その後 upstream の Strata 0.2.22（エンジンの版は 0.1.69、`b28ca95`）までの変更のうち、
この fork に当たるものを取り込んでいます（[Strata 0.1.69 の取り込み](#strata-0169-の取り込み)から [Strata 0.1.38 の取り込み](#strata-0138-の取り込み)までの各節）。
同じ SYCL のコードを intel/llvm で NVIDIA の GPU 向けにもコンパイルします（contrib-llvm のビルド、[ビルド](BUILD.ja.md#ビルド)）。
upstream の CUDA のビルドは廃止してソースも消しました。
CUDA のソースは upstream の Strata（`3ce2523`）に残っていて、Xe の各ソースは移植元の CUDA のファイル名をコメントに書いています。

開発と測定は Intel Arc Pro B70（Xe2、32 GB）で行っています。
記録にある速さは、断りがなければこの GPU のものです。NVIDIA は RTX 4070 と RTX 3070 で確かめています。
エンジンは経路を GPU が報告する能力で選びます（AGENTS.md の最初の規則）。
プリフィルの行列積は Xe2 の XMX には専用のカーネル（xmx_gemm）を、Arc A シリーズの XMX と NVIDIA の Tensor Core には
報告されたタイルの形で動く joint_matrix の積（mma_gemm）を、RDNA3・RDNA4 には同じ積を WMMA の命令で動かしたものを使います。
一方、行列エンジンのない GPU は DP4a の経路を使います。
量子化したまま int8 で積むエキスパートの積（iq_mmq）は、Intel の XMX（Xe2・Xe-HPG）、NVIDIA の Tensor Core（Turing 以降）、RDNA3・RDNA4 の WMMA で動き、DP4a で積むのは行列エンジンのない GPU だけです。
int8 の行列エンジンのない Volta は FP16 の経路を使います。
Qwen3.8 の prompt は xmx_gemm が動く GPU では FP16 の経路のままにします（B70 ではそちらが速い）。

小さい GPU は B70 を小さく見せて確かめます。
`STRATA_VRAM_LIMIT_MIB` はエンジンから見える VRAM を、`STRATA_MAX_ALLOC_MIB` は想定する最大の確保量を制限し、
`STRATA_NO_XMX=1` は XMX を使わない経路を選ばせます。
CPU 内蔵のグラフィックス（開発機では UHD 770）は確認用の 2 つ目の実機として使います。

## 890M の decode の見直し

Qwen3.8-Flash-Next の decode の 1 窓を段ごとに測り、段ごとに速くなる候補を挙げて、見込みの大きい順に 1 つずつ実装して測りました（2026-11-16 と 17）。
入れた変更は 31 件です。890M の greedy の 6 チャット（IQ2_XS、`--spec 4`、`--max-context 32768`、交互 3 周）の decode は、36.47・36.45・35.91 tok/s から 42.15・42.66・42.19 tok/s になりました（平均 36.28 から 42.33、+16.7%）。答えは変更の前と同じです。
どの変更も、装置の報告（統合 GPU、HIP のバックエンド）から選ぶか、どの GPU でも同じビットになる形で入れています。

### 1 窓の時間の内訳

変更の前の 1 窓は 76.4 ms で、平均 2.64 トークンが出ていました（6 チャットの 433 窓の平均、段ごとの時計を入れた計測）。

| 段 | 1 窓の時間 | 割合 | 主な内訳 |
| --- | --- | --- | --- |
| エキスパート | 22.3 ms | 29.2% | エキスパートの積 18.8、shared と量子化 2.9 |
| GDN（36 層） | 18.9 ms | 24.7% | qkv の GEMV 6.0、out-proj 4.1、z 4.1、再帰 3.9 |
| ハイパーコネクションの読み出し | 18.6 ms | 24.3% | 後半の読み出しと router 10.0、前半の読み出し 8.5 |
| QSA の注意（12 層） | 5.9 ms | 7.7% | q と索引の query の GEMV 2.8、out-proj 1.4 |
| 推測デコードの下書き | 5.6 ms | 7.4% | MTP の下書き |
| 出力ヘッド | 3.9 ms | 5.0% | 338 MB の IQ4_XS の積 |
| GPU の待ち | 1.7 ms | 2.2% | host を待つカーネルと、空の起動 |
| host | 0.4 ms | 0.5% | 層ごとの計画、commit と出力 |

段ごとの時計を入れると窓は遅くなり、時計なしの 1 窓は約 72 ms です。
カーネルの起動の間の隙間は、段の時間に入っています。層ごとに空の起動を足して測ると、起動 1 つの費用は約 2 us でした（1 窓に 576 起動を足して −1.5%）。

### 入れた変更

数字は 890M の greedy の 6 チャットの decode で、その変更の前後を交互に 3 周（断りのあるものは 4 周）測った平均です。答えはどれも前後で同じです。

- 全エキスパートが cache にあるときの窓グラフ: `STRATA_ADAPT_LAG` の既定が 2 になってから（2026-10-25 の merge）、全エキスパートが cache にあっても、窓は層ごとに host が計画するグラフで動いていました。
  入れ替えの写しを待たない設定で止める必要があるのは、装置が層ごとに居場所の表を読む計画（`STRATA_VERIFY_DEVICE_PLAN`）だけです。host に聞かない窓グラフは、窓ごとに host の表を見て選ぶので、この設定でも使うようにしました。
  層ごとの GPU と host の往復と、host を待つカーネルがなくなります。36.01 → 36.58 tok/s（+1.6%）。
- 側枝: 窓グラフの側枝（GDN の層の ab と z、shared expert をほかのカーネルと並べる）を使うかどうかは、起動時に小さいカーネルのグラフで測って決めています。ROCm 7.2 ではこの計測の差が 1〜4% になって 5% の線に届かず、側枝を使っていませんでした。
  HIP の統合 GPU では、host に聞かない窓グラフは計測の結果にかかわらず側枝を使います（`STRATA_DF_BRANCH=0` で切れます）。側枝が窓にもたらすのはエキスパートのメモリー律速のカーネルとの重なりで、起動時の計測のグラフにはそれがないためです。
  host が計画するグラフは、これまでどおり計測で決めます（890M では、側枝を入れると 36.44 → 36.22 tok/s に下がります）。36.78 → 37.72 tok/s（+2.6%）。
- ROCm のランタイムの待ち方: AMD の GPU が統合 GPU だけの機械では、エンジンが装置を列挙する前に `ROC_CPU_WAIT_FOR_SIGNAL=1` を環境に置きます（利用者が設定していればその値のままです。Linux だけ）。
  ROCm のランタイムは、別のキューのコマンドを待つとき、既定では GPU の側で待つバリアを積みます。この設定では CPU の側で待ちます。側枝は層ごとにキューを分けて合流させるので、合流のたびにこの待ちが入ります。
  37.72 → 38.10 tok/s（+1.0%）。2,669 トークンの prompt の prefill は 329.8 → 329.1 tok/s で、差は揺れの中です。
- 推測デコードの費用の表: `STRATA_SPEC_COST` が幅ごとに持つ 1 周の時間が外れたまま戻らず、窓が狭いほうか広いほうに寄ったままプロセスの終わりまで続く起動がありました（34.0〜35.6 tok/s、およそ 60 回の起動のうち 4 回）。
  幅ごとの最初の 3 回は、平均ではなく最小値で数えます。その後は、平均より短い時間を重み 0.25 で、長い時間を 0.1 で追い、長い時間は平均の 1.25 倍で切ります。MTP の連鎖は、128 周より長く測っていない幅を、下書きが続く見込みが 0.5 以上のときにもう一度試します。
  変更の後の 14 回の起動には、表が外れた回がありませんでした。`STRATA_DECODE_TIMING=1` は、要求ごとに窓の幅ごとの回数と費用の表も 1 行出します。
- `--lookup-chain 2`: 費用の規則が既定の装置（HIP の統合 GPU）では、`--lookup-chain 2` を既定にしました。`--mtp` と suffix の下書きがあり、`--batch`、パイプラインの窓、`--draft dflash2` のどれでもないときです（`--lookup-chain 0` で切れます）。
  足す行の費用は DraftPolicy が測った時間から見積もり、割に合うときだけ足します。窓の上限は 6 行から 8 行になり、verify の装置のバッファは 72.5 MiB から 86.6 MiB に増えます。38.03 → 38.31 tok/s（+0.7%、4 周）。
- combine が読むゼロの行: host に聞かない窓グラフには CPU が計算する分がありません。層ごとにゼロで埋めていた CPU の分の行（1 トークンあたり 10 行）を書くのをやめ、combine は GPU が計算した行だけを読みます。結果はビット単位で同じです（`native_multi_parity`）。
- QSA の索引の点数の格子: 点数のカーネルは、`--max-context` から決めた格子（32768 では層ごとに 1,025 ワークグループ）を起動し、文脈より先のワークグループはすぐ戻っていました。
  計算ユニットの数から決めた固定の格子（890M では 64 ワークグループ）にし、サブグループが格子の幅ずつ先のブロックへ進みます。点数はビット単位で同じです（`qsa_score_parity`）。
- 文脈が選択の幅より短い間の索引: 文脈が 2,051 セル以下の間、選択は全セルで、点数を読みません。その間は、索引の query 側の積（層ごとに重み 2.6 MB）と点数の計算を飛ばします（`STRATA_QSA_SKIP_IDENTITY=0` で元の動き）。

最後の 3 つは、合わせて 38.20 → 38.66 tok/s（+1.2%）です。

ここまでの 8 つを入れた形で段ごとの時間を測り直し、候補をさらに 2 回探して、次の 8 つを入れました。

- shared expert の側枝を分ける位置: 側枝は router、top-10、計画のあとで分けていました。shared expert が読むのは層の入力だけなので、router の前で分けます（`STRATA_SH_EARLY=0` で元の位置）。
  router の GEMV、top-10、計画の間はメモリーの帯域が空いていて、shared expert の重み（層ごとに約 2.7 MB）をその間に読めます。変わるのは起動の順だけです。38.82 → 38.92 tok/s（+0.3%、冷えた状態から始まった 1 周を除く 2 周）。
- 窓が確かめない下書きの PLE の行: 下書きの層は、下書きごとに PLE の行を先読みします。費用の規則が最後の下書きを窓に入れなかったときは、その行を読みません（`STRATA_PLE_SKIP_DROPPED=0` で元の動き）。
  窓の起動は出ている読みをすべて待つので、起動の前の待ちが 1 窓 0.21 ms から 0.16 ms に減ります。
- PLE の射影: 層 1 の PLE の key と value の射影（BF16 の重み 65.5 MB）は、層 1 の始めに直列で動いていました。host に聞かない窓グラフでは、窓の始めに 3 本目のキューで起動し、層 1 で合流します（`STRATA_PLE_BRANCH=0` で元の位置）。
  同じカーネルを同じ入力で動かします。decode の差は揺れの中で、1 窓の時間が 0.03〜0.04 ms 短くなりました。
- MTP の窓の上限: 費用の規則が既定の装置では、`--mtp-max-t` を指定しないときの MTP の窓の上限を `--spec` より 1 行広くしました（`--spec 4` なら 5 行。`--mtp-max-t 4` で元の上限）。
  1 窓のトークンは 2.62 から 2.72 に増え、38.78 → 38.97 tok/s（+0.5%）。上限を 6 行にしても速くならなかったので（38.75 → 38.73 tok/s）、広げるのは 1 行だけです。
- PLE の表の読みのスレッド: 統合 GPU と HIP では、表の直接読みをトークンのスレッドが出して回収します（`--ple-io-thread` で I/O のスレッドに戻ります）。I/O のスレッドを通すと、窓の起動を待たせる最後の読みに、スレッドの受け渡しが 2 回余分に入るためです。
  起動の前の待ちは 1 窓 0.21 ms から 0.12 ms になり、38.81 → 38.87 tok/s。SSD の keep-alive は、この形でも専用のスレッドが続けます。
- 下書きの層の正規化: 統合 GPU と HIP では、`--mtp-hnorm stream`（下書きの層の隠れ入力を、ハイパーコネクションのストリームごとに正規化する）をエンジンの既定にしました（`--mtp-hnorm pooled` で元の形）。
  setup は 2026-10-29 から統合 GPU の設定にこのフラグを書いていますが、それより前に作った設定や手で書いた設定は pooled のままで動いていました。この計測に使ってきた設定もそうでした。1 窓のトークンは 2.70 から 2.91 に増え、39.04 → 40.52 tok/s（+3.8%）。
- GDN の step のバリア: 窓の GDN の step は、窓の行ごとにバリアを 2 つ置いていました。和の置き場は行ごとに別なので、出力の和を行のループのあとにまとめ、行ごとのバリアを 1 つにしました。足す順は同じで、結果はビット単位で同じです（`gdn_step_parity`）。
  カーネル単体で 3 行の step が 73.2 → 62.1 us、decode は 40.60 → 40.79 tok/s（+0.5%）。
- 窓の幅の選び方: 費用の規則は、窓ごとに「見込みのトークン ÷ 1 周の時間」がいちばん大きい幅を選んでいました。窓の列の全体の率を λ として、「見込みのトークン − λ × 1 周の時間」がいちばん大きい幅を選ぶ形にしました（`STRATA_SPEC_RATE=0` で元の選び方）。
  確かでない下書きの窓は狭く、確かな下書きが続いた窓は広くなります。採択される下書きの数は同じで、確かめる下書きが減り、40.51 → 40.64 tok/s（+0.3%、4 周）。

あとの 8 つを入れた形は、前の 8 つだけの形と交互に 3 周測って、38.82 → 41.01 tok/s（+5.6%）でした。

そのあと、主なカーネルの gfx1150 の ISA をソースと突き合わせて、ソースが求めていない待ちや読みを探し、次の 7 つを入れました。どれも結果はビット単位で同じです。

- ハイパーコネクションの読みの up: lane が受け持つ 2 行のそれぞれで、窓の各トークンの lo（float4 が 10 個）を local memory から読んでいました。1 行目の結果を書く store が間にあり、コンパイラーは 2 行目の読みを省けなかったためです。トークンごとに 1 回読んで、2 行に使います。
  local memory の読みは 4 行の窓で 112 回から 72 回になり、カーネル単体で 3 行の読みが 103.6 → 99.2 us、decode は 40.49 → 40.89 tok/s（+1.0%、4 周）。
- IQ2_S の符号のマスク: 符号のビットからバイトのマスクを作る式を、コンパイラーが 32 bit の乗算（gfx11 では 4 分の 1 の速さ）に戻していました。gate と up の行の 1 歩に 8 個です。AMD の device compile では、引き算と xor で作ります。
  カーネル単体で 3 行の窓のエキスパートが 441.2 → 438.1 us。
- エキスパートの行のアドレス: 統合 GPU と HIP の行は、歩ごとに、ブロックとサブブロックを割り算と剰余で、位置を乗算で出していました。lane のサブブロックはどの歩でも同じなので、lane ごとに 1 回計算し、歩は定数の差にしました。
  カーネル単体で、gate と up の行の変更が 438.2 → 432.8 us、down の行の変更がさらに 432.2 → 429.3 us。decode は 41.52 → 41.92 tok/s（+1.0%、4 周）。
- ハイパーコネクションの読みの down: 行ごとの和を sub-group に配る所で、配列の 2 つの要素の間の条件式が、16 要素の比較と選択になっていました。2 つの値を先に取ってから選びます。カーネル単体で 4 行の読みが 102.9 → 102.2 us。
- GDN の量子化: 閉じの norm と量子化のカーネルで、sub-group に集める 2 つの reduction のループが展開されず、交換の命令を選ぶ分岐が実行時に残っていました。ループを展開し、カーネルは 537 行から 366 行になりました。
- エキスパートのバリア: gate と up、down のカーネルで、データの流れが求めていないバリア（エントリーの組の最初のもの）を、統合 GPU と HIP のカーネルでは置きません。
- 層ごとの計画: 全エキスパートが cache にある窓の計画のカーネルで、エキスパートの番号を 2 回読んでいた所を 1 回の走査にし、合計をもう一度集める reduction をなくしました。

この 7 つを入れた形は、その前の 16 件の形と交互に 3 周測って、40.99 → 41.92 tok/s（+2.3%）でした。

続けて、本線に直列に並ぶ小さい起動と、カーネルの中の待ちを減らす、次の 8 つを入れました。どれも結果はビット単位で同じです。

- ハイパーコネクションの読みの up の表: 融合カーネルの up のグループは、トークンごとの表を local memory に写すためだけに、バリアを 1 つ取っていました。写しを、行のロードを待つ間に行います。カーネル単体で 4 行の読みが 103.4 → 102.4 us。
- GDN の step の k: 行ごとに、k の 32 個を local memory から 2 回（バリアの前とあと）、2 個ずつ読んでいました。float4 で 1 回だけ読み、バリアをまたいでレジスターに持ちます。そのためのレジスターは、状態の書き戻しのアドレスを作り直して空けました。
  VGPR は 126 本から 104 本になり、カーネル単体で 3 行の step が 61.8 → 60.1 us、保留 3 行つきの step が 71.0 → 64.1 us、decode は 41.80 → 42.11 tok/s（+0.7%、4 周）。
- BF16 の GEMV（router、PLE、索引の射影）: 和の枠を 0 で埋めるためだけのバリアをなくしました。0 を書くのは、sub-group が書かない枠だけです。decode は、上の GDN の変更と合わせて測りました。
- router の top-10: 全エキスパートが cache にある窓では、top-10 を層ごとの計画の起動の中で行います（`STRATA_PLAN_FUSE=0` で元の 2 つの起動）。decode は 42.24 → 42.52 tok/s（+0.7%、4 周）。
  減るのは層ごとに起動 1 つです。空の起動を足して測った費用（1 つ約 2 us）からの見込みは +0.14% でした。本線に直列に並ぶ小さい起動は、空の起動より高くつきます。
- エキスパートの入力の量子化: 上の形の層では、層の入力の q8_1 を、計画の起動の追加の work-group で作ります。shared expert の側枝は読みの直後に分かれ、自分のキューで同じ像を作ります（`STRATA_Q8_PLAN=0` で元の形）。decode は 42.48 → 42.53 tok/s で、差は揺れの中です。
- 注意の gate と量子化: QSA の層の merge、gate、q8_1 の量子化の 3 つの起動を、1 つにしました（`STRATA_QSA_MERGE_FUSE=0` で元の形）。decode は 42.53 → 42.71 tok/s（+0.4%、4 周）。
- MTP の下書きの層: 埋め込みと隠れ入力の norm が q8_1 も書き、K の norm と RoPE を 1 つの起動で行い、最後の残差の書き込みを最後の読みに畳みます。検証の窓の最後の書き込みも、head の読みに畳みます（`STRATA_MTP_GLUE_FUSE=0` で元の形）。decode は 42.61 → 42.68 tok/s で、差は揺れの中です。
- 検証の窓の始めと終わり: 窓の入力（トークン、step、位置、PLE の行）の写しを 1 つの起動にまとめました。greedy の選択のカーネルは、ロジットに NaN や無限大がないかの検査も行います。検査だけの起動が同じロジットをもう一度読むことがなくなります（`STRATA_WINDOW_GLUE_FUSE=0` で元の形）。decode は 42.71 tok/s のままで、差はありません。

この 8 つを入れた形は、その前の 23 件の形と交互に 3 周測って、41.35 → 42.33 tok/s（+2.4%）でした。

### 入れなかった変更

- host に聞かない窓グラフで PLE の読みを起動のあとに行う形は、37.25 → 37.01 tok/s と遅くなりました。
- z の GEMV を qkv のあとで分けて GDN の再帰と重ねる形は、37.79 → 37.29 tok/s（−1.3%）でした。
- エキスパートの重みを 32 bit と 64 bit の広いロードで読む形は、コンパイラーが 16 bit の読みをすでにまとめていて、ロードの命令の数が変わりませんでした。変わったのは non-temporal の hint だけで、37.29 → 36.89 tok/s（−1.1%）でした。
- GDN の再帰をレーンごとに計算する形は、カーネル単体で 3 行の step が 74.5 → 110.4 us と遅く、起動時の選択も一度も選びませんでした。
- GDN の step を、1 レーンが 1 列の状態を持つ形にする案は、既存の形とビット単位で同じ答えでしたが、カーネル単体で 3 行の step が 72.7 → 144.5 us と遅くなりました。コンパイラーがこのカーネルに VGPR を 256 本使い、同時に走るウェーブが減るためです。
- 保留の行のあとの GDN の状態の書き戻しを、窓の行の計算と重ねる形は、カーネル単体で差がありませんでした（保留 3 行つきの 3 行の step が 110.8 us と 114.8 us）。
- `GPU_MAX_HW_QUEUES=8` は 38.75 → 37.12 tok/s（−4.2%）、`ROC_ACTIVE_WAIT_TIMEOUT=50` は 38.75 → 38.80 tok/s で差がなく、どちらも入れていません。
- Q2_0 の down の行のロードから non-temporal の指定を外す形は、39.07 → 39.05 tok/s で差がありませんでした。
- ROCm のランタイムの設定（`DEBUG_HIP_DYNAMIC_QUEUES=0`、`DEBUG_HIP_FORCE_GRAPH_QUEUES`、`DEBUG_CLR_GRAPH_PACKET_CAPTURE`、`DEBUG_CLR_BATCH_CPU_SYNC_SIZE=1`）と `STRATA_WAIT_SLEEP_US=20` は、どれも 40.7〜41.3 tok/s の揺れの中でした。窓のキューの優先度を上げる形は −0.6% でした。
- エキスパートの行では、down の行の読みを 6 から 5 にする形（3 行の窓で 440.9 → 443.2 us）、gate と up のワークグループを 64 行にする形（4 行の窓まで遅く、速いのは 6 行だけ）、エントリーの判定をループの break にする形（431.0 → 645.8 us）が、どれも速くなりませんでした。
- GDN の step の書き戻しを別のポインターで行う形は、レジスターが 126 本から 64 本に減りましたが、それだけでは、カーネル単体でも decode でも差がありませんでした。空いたレジスターは、k を持つ形（上）で使っています。
- GDN の step の閉じの norm を、再帰の起動の追加のグループにする形は、カーネル単体で 3 行の step が 61.8 → 56.4 us でしたが、decode は 42.61 tok/s と 42.60 tok/s で差がありませんでした。グループが前の段を spin で待つ形は、カーネル単体の差が decode に出ません。同じ形になる、router の GEMV と計画の融合なども作っていません。
- ハイパーコネクションの読みの up で、x や partials、lo のロードを先にまとめて出す 4 つの形は、カーネル単体で差がありませんでした（3 行の読みが 97.2 us と 97.5〜98.3 us）。
- 注意の query を lane ごとに並べる形は、カーネルの ISA が 5〜10% 短くなりましたが、decode は 42.11 → 41.60 tok/s（−1.2%）でした。
- 注意のカーネルで q と merge の読みをまとめて出す形は、decode が 4 周とも下でした（差は 0.02〜0.49 tok/s）。ハイパーコネクションの読みで inject のブロックを先に出す形は、差がありませんでした。
- 精度を落とす候補（出力ヘッドを 2 段にする、ハイパーコネクションの重みを 8 ビット未満にする、router を Q8_0 にする、など 7 件）は試していません。カーネルが CPU の参照と合うという確かめ方に当てはまらないためです。

確かめたのは、free と contrib-llvm のビルド、CTest（どちらも pack のない 2 本を除いて通ります）、`tools/lint/run.sh` に新しい指摘がないことです。

## Strata-rdna3 からの移植

[nicejunjie/Strata-rdna3](https://github.com/nicejunjie/Strata-rdna3)（RX 7900 XTX 向けの HIP のフォーク）から、890M の prefill を速くする 3 つの変更を移植しました（2026-11-18）。
本体は gfx1150 のデバイスコードだけにあり、ホストはデバイスが gfx1150 と報告したときだけ使います。ほかの GPU の経路は変わりません。

| 変更 | 何をするか | 既定 | prefill（3,461〜30,153 トークン） |
| --- | --- | --- | --- |
| `STRATA_PA_FAST` | QSA の prompt attention で、q と p を FP16 の 2 枚（hi と lo）ではなく 1 枚で持ち、q.k と p.v の WMMA を半分にする | gfx1150 でオン（`=0` でオフ） | 2.4〜3.2% 速い |
| `STRATA_HC_UPMIX` | ハイパーコネクションの読みの up 投影と mix を 1 つのカーネルにし、途中の行列（16K トークンで 671 MB）を書かない | gfx1150 でオン（`=0` でオフ） | 1.7〜3.2% 速い |
| `STRATA_SELECT_WMMA` | QSA のブロック選択の点数を WMMA で求める（BF16 の 3 分割で 3 回の積、`=3` で 6 回） | オフ（`=1` でオン） | 揺れの中 |

890M（IQ2_XS、greedy、docs/DETAILS.ja.md の先頭を要約、交互に 3 周）で、PA_FAST と HC_UPMIX を両方使うと、prefill は 3 周とも 3.9〜5.0% 速くなりました。

| トークン | 両方オフ（ms） | 両方オン（ms） |
| --- | --- | --- |
| 3,461 | 10,670 / 10,663 / 10,667 | 10,226 / 10,209 / 10,254 |
| 8,145 | 25,310 / 25,338 / 25,345 | 24,138 / 24,150 / 24,155 |
| 16,050 | 50,659 / 50,727 / 50,746 | 48,371 / 48,338 / 48,488 |
| 30,153 | 96,491 / 96,517 / 96,786 | 91,932 / 91,884 / 91,982 |

- HC_UPMIX は答えを変えません。
- PA_FAST は FP16 で丸めるので、一部の長さで答えの言い回しが変わります（内容は同じです）。prompt のあとの decode も、MTP の下書きの当たり方が変わるので文書によって上下します。8 つの文書の要約（256 トークン）では、7 つで −2.2〜+10.7%、幾何平均 +1.7% でした。
- SELECT_WMMA は、点数のカーネルだけなら 8K で 1.39 倍速くなりますが、ブロック選択は prefill の 1% に満たないので、全体の速さは変わりませんでした。
- 試験は `qsa_prompt_attn_parity_pa_fast`、`qsa_prompt_attn_parity_split`、`qsa_select_wmma_parity`（`_p3`）、`gr_upmix_parity` です。

移植しなかったものは、次のとおりです。

- ランタイムのホストブロックの mlock: この機の SYCL には、rdna3 が探す `/dev/zero` の共有マップがありません。
- hc write の 16 バイトの読み書き: 見込みが 0.4% ほどです。
- エキスパートの prefill を FP16 にほどいて積む形: 890M のモデルの量子化形式には専用のコードブックが要り、780M では差がありませんでした。
- decode の細かいカーネルの切り替え: 890M の decode は帯域で決まり、見込みが小さいと判断しました。

## 上流の XeStrata の取り込み（2026-11-25 の main まで）

上流の XeStrata（MistVVK/XeStrata）の main を、xe0.2.8-1 の次から 2026-11-25 の aadd4f9fe（xe0.2.20-1 の 25 件先）まで merge しました（2026-11-27）。
5,327 コミットのうち、Strata の取り込みではない上流自身のものは 1,411 件です。
上流は Strata を 0.2.20 まで取り込んでいて、この fork は 0.2.22 まで自分で移してあります。同じ変更が両側にある所は fork の形を残し、上流がその上に足した直しを移しました。
衝突したのは 189 ファイルの 1,111 か所です。

入ったものは次のとおりです。

- 名前: 実行ファイルは `xestrata-engine`、`xestrata-image`、`xestrata-device` になりました（serve の `--engine strata` は引き続き使えます）。環境変数は `XESTRATA_<名前>` で、エンジンは `XESTRATA_<名前>`、`STRATA_<名前>` の順に読みます。ログの接頭辞は `xestrata` です。この fork が足したスイッチの名前は変わりません。
- エンジンの構造: verify の窓の MoE を 1 つにまとめた `MoeWindow`、prompt の注意の形を選ぶ `flash_attn`、Qwen3.8 と密なモデルが共有する読み方（QSA、GDN、PLE、BF16 の積、埋め込み）、環境変数の読み取り `strata::env_get`、行列の命令の形の報告（`has_form`）。
- decode: 返答の途中のチェックポイント（`--reply-checkpoint-every`、既定は 512 トークンごと）、GPU を失ったことの検出（verify と MTP の待ちが event を読み、20 秒で打ち切ります）、16 行までの窓（`--wide-window`。MTP の窓は 8 行のままです）、起動時に MTP、DFlash2、下書きなしを測って選ぶ `--draft auto`（DFlash2 の pack が要ります）、統合 GPU が全エキスパートを持つときにモデルのファイルのページキャッシュを返すこと（`XESTRATA_UMA_DROP_PAGES=0` で残します）。
- prefill とカーネル: 密な積を形ごとに自前のカーネルとライブラリーで比べて選ぶこと、RDNA の WMMA のタイルの形、エキスパートの int8 の積の本体（K の塊のループをなくしてレジスターの退避を減らしたもの）、エキスパートの積の FP16 の形（`XESTRATA_IQ_MMQ_F16`）、MoE のモデルの密な射影の W8A8（`XESTRATA_LLM_W8A8=1`、opt-in）、上流の GDN の塊の走査（opt-in）、aotriton 0.15b の注意と comfy-kitchen の int8 の積のモジュール（AMD の GPU でだけ開きます）。
- ほかのモデル: Inkling-2-Mini をエンジンの中で動かす経路（`src/inkling2`）、画像の Qwen-Image-2.1、MOSS-Image の編集、Anima、ControlNet の前処理、LoRA の学習、RealESRGAN と SeedVR2 の拡大、SAM 3、BiRefNet、Florence-2、動画（ffmpeg を呼びます）。
- serve と web ページ: 起動のたびに決める `auto` の設定（`serve/autotune.py`）、画像のモデルと同じ GPU で文のエンジンを降ろすときの共有の arena、答えにほかの言語の文字を出さない `script_guard`、会話の保存、モニターの GPU の役割のタブ。web ページは上流のものになり、チャットの一覧、編集、数式が既定で入ります。
- setup: `--drafting auto|mtp|dflash2|off`、AMD の ROCm の確認（ライブラリーの名前で見ます）、画像と翻訳のモデルの質問、設定の一覧（`data/settings-register.json`、`tools/settings_register.py` が作ります）。
- intel/llvm: `tools/intel_llvm_build.py` は v7.2.1 を指し、パッチは 31 本です。HIP の仮想メモリー、hostcall を要らなくする AMDGPU の修正、モジュールの読み込みでのメモリー不足の報告が加わりました。この fork の `hip-reqd-work-group-size` は 33 番に移しました。エンジンは実行時に GPU へ仮想メモリーの対応を尋ねるので、v7.2.0 で作った intel/llvm のままでもビルドでき、これまでどおり動きます。

fork の形を残したものは次のとおりです。HIP の統合 GPU であることは、デバイスの報告から決めます。

- 推測デコード: 費用の規則（`SPEC_COST`）は HIP の統合 GPU ですべての深さで入ります。上流の `auto`（6K から）は入れていません。DFlash2 は opt-in のままで、HIP の統合 GPU では fork の注意（`XESTRATA_DF2_ATTN=4`）、融合した Q4_0 の積、`--draft-conf 0.6` を使います。setup は統合 GPU では `--drafting mtp` を既定にします。
- verify の窓: 統合 GPU での横の分岐、top-10 を plan の起動に入れる形、GDN の段階分けと後回しの commit、`STRATA_VERIFY_ECAP`、`STRATA_EXPERT_KEEP` と `STRATA_EXPERT_SER`。これらは `MoeWindow` の上に載せました。
- decode のカーネル: 注意の per-cell の形と GEMV の行の配置の計測は、HIP の統合 GPU では行いません（`STRATA_QSA_PERCELL=1`、`STRATA_MMVQ_TUNE` で試せます）。`STRATA_QSA_CU_SLOTS` は opt-in のままです。`GPU_MAX_HW_QUEUES=2` は統合 GPU には設定しません。Gumbel の標本化の腕は、統合 GPU では `XESTRATA_SPEC_GUMBEL=auto` のときだけ使います。
- prefill: `DEQUANT_SIDE`（上流は消しました）、prompt の注意の `PA_FAST`（fork の `qsa_prompt_attn.cpp`）、GDN の WMMA の再帰（`GDN_CHUNKED`。上流の同じ名前の走査は `=1` のときだけで、WMMA の経路が動かない所に限ります）、`STRATA_GDN_TC`（opt-in）、`PF_DM16`、`HC_UPMIX`、KVA の identity だけの形、起動時に rocBLAS と自前のカーネルを比べる試し、塊 2048 と layer-major の off。
- CPU: WAITPKG のない CPU では pool の待ちはこれまでどおり spin で、MWAITX は `POOL_UMWAIT=1` のときです。IQ の gather は CPUID で決めます。
- serve と setup: エンジンの nice は 0 のまま（`XESTRATA_ENGINE_NICE=auto` で上流の +5）、`--vram-elastic` は統合 GPU では opt-in のまま、道具呼び出しの解析器は Strata 0.2.22 の形、統合 GPU の設定（`uma_host_cache`、`kfd_integrated`、`--mmap-experts`、`STRATA_CACHE_HOST=1`）と最初の提案（IQ3_XXS か Q2_0）、Unsloth の一覧。
- README とこの fork の文書、`container/`（Fedora 44 と Fedora の ROCm。`XESTRATA_ROCM=10.1` で AMD の ROCm）、CI。

この fork が自分で持っていた Inkling-2-Mini のエンジン（`src/inkling2mini/` の専用のプログラム）は、この取り込みのあとで外しました（2026-11-27）。
エンジンのソースと試験、serve と setup の専用の分岐（`--family inkling-2-mini`）、`.strata` のファイルを読むために持っていた Zstandard の decoder（`third_party/main/zstd-dec/`）がなくなり、Inkling-2-Mini の経路は上流のもの（`src/inkling2`、setup の `inkling` と `inkling-qsa`）だけになります。
この fork の目標は Qwen3.8-Flash-Next で、上流と同じファイルだけを持つほうが、次の取り込みで衝突が減るためです。

既定が変わったものは次のとおりです。

- CPU の pool は、down の仕事が自分のエキスパートの gate と up だけを待つ形になりました（`XESTRATA_POOL_DEP=0` で 2 段の形）。答えのビットは同じです。
- AVX-512 のある CPU では、IQ4_XS の gate と up を 512 ビットの行で読みます（`XESTRATA_IQ512_IQ4XS=0` で切れます）。
- setup が入れる画像の読み手は、エンジンの `xestrata-vision` になりました。この fork の mtmd の読み手（`tools/vision`）はソースからビルドできます。

890M に合わせて直したものは次のとおりです。

- エキスパートと密な射影の int8 の積（`iq_mmq.cpp`）: 上流は、行の少ない積で K を 2 つに分けて別々の work-group に渡す経路を足しました。その判定と書き出しが RDNA の本体に入ったことで、累算器がレジスターに載らずスタックに置かれ、密な射影の 128 行のタイルが 890M で 2 倍近く遅くなっていました（`iq_mmq_parity --dense`、IQ3_S の 10240 x 2560、2,048 行で 14.8 ms が 27.3 ms。prefill は 3,426 トークンで 340 tok/s が 275 tok/s）。K を分ける経路は 64 行までの積にしか使われないので、それより大きいタイルの本体からは外し、小さいタイルでは累算器のループを展開するよう指定し、書き出しの行き先を 1 回だけ調べる形にしました。gfx1150 では本体の写しを 1 つにしています（上流の 2 つの写しでは、IQ3_XXS の 64 行以上の積が 12% 遅くなりました）。直したあとは、密な積が 12.9 ms 対 13.2 ms（IQ4_XS、2,048 行）、エキスパートの積は merge 前と 1% 以内です。
- MTP の下書きの待ち: 上流は、GPU を失ったことを知るために起動のたびに event を記録します。HIP の統合 GPU では、待ちが 200 ms を超えたときにだけ記録します。1 段は数 ms で終わるので、そこまで待った時点で止まっていると見てよく、失われたキューは遅れた記録も断ります。
- 上流の DFlash2 の注意のカーネル（256 work-item の形）の不具合: 文脈の鍵が塊の数より少ないとき、鍵のない塊が番兵を一部しか書かず、和に残りの値が混ざっていました（`dflash2_attention_parity` の短い文脈の 3 例が inf）。番兵をサブグループごとに全部書くようにしました。

確かめたこと: free と contrib-llvm のビルド、CTest（どちらのモードも 259 件が通ります）、serve の Python の試験（1,904 本、28 本は skip）、tools の試験（736 本、29 本は skip）、`reuse lint` と `tools/license_class.py`、Python、Markdown、CSS の lint で merge が新しく足した指摘がないこと。

890M での merge 前（5e04b832c）との比較は次のとおりです（IQ2_XS、`--spec 4`、どれも merge 前、merge 後の順）。

- decode: 6 チャットの greedy の答えは 4 周とも同じでした。速さは 3 周の平均で 42.72 対 42.65 tok/s です。
- prefill: 3,426 トークンで 340.5 対 340.4 tok/s、14,210 トークンで 335.0 対 335.3 tok/s です。
- 長い prompt のあとの答え: 14,210 トークンの prompt では、要約の文面が 27 文字目から変わりました。どちらの文面も毎回同じに出ます。最初の 3 層は 1 ビットも違わず、最初の注意の層で 511 行のうち数行に 1e-4 程度の差が出て、後ろの層で経路の選択が変わります。
- 普段の設定に近い形（IQ3_XXS、`--spec 6`、文脈 262,144）: 6 チャットの答えは 3 周とも同じで、decode は 3 周の平均で 33.74 対 33.95 tok/s です。3,426 トークンの prefill は、どちらも 274〜309 tok/s の間で回ごとに動きました。

## 上流の XeStrata の取り込み（xe0.2.8-1 まで）

上流の XeStrata（MistVVK/XeStrata）の main を、xe0.2.2-1 の次から 2026-10-30 の 721afe51f（xe0.2.8-1）まで merge しました（2026-11-02）。
1,645 コミットのうち、Strata の取り込みではない上流自身のものは 491 件で、そのうち 108 件は Strata 0.2.6〜0.2.8 の変更を SYCL に移したものです。この fork は Strata 0.2.9 まで自分で移してあるので、同じ変更が両側にある所は fork の形を残しました。

入ったものは次のとおりです。

- 画像のエンジン: HunyuanImage 3.0、MOSS-Image、MageTrail と Flux2 VAE、LoRA（LoKr、DoRA、LoHa、OFT/BOFT、GLoRA を含む）、img2img とマスク、途中の画像、RDNA3 の int8 の注意。setup の `--images` で複数のモデルを入れられます。890M では Mage-Flow-Turbo（bf16、DiT は読みながら 8 ビットに）で 1024 x 1024 の 1 枚を 4 ステップ 29.9 秒で作れました（文の読み取り 4.1 秒、サンプリング 23.8 秒。AOTriton の注意のモジュールはないので自前のカーネルです）。ほかの画像のモデルは動かしていません。
- serve: OpenAI と Anthropic の API の抜けを埋める一連の修正（`/v1/responses` の tool_choice と allowed_tools、Chat の `n` と古い `functions`、Anthropic の構造化出力、PDF とテキストのファイルの読み取り、エラーの形の統一）、strict の道具とスキーマの文法での縛り（XGrammar、`STRATA_GRAMMAR=0` で切る）、Project Maya のループ対策（`loop_guard`、既定で入る）、モデルの思考の段を `/health` で知らせる仕組み、SIGHUP での後始末。
- web ページ: 画像のタブの拡張、言語の既定をブラウザーから選ぶこと、XeStrata だけの画面の文の 9 言語の訳、min_p の欄。
- エンジン: TQP5〜TQP7（Gyro-Coder）の型、エキスパートの int8 の積の Q5_1、2 ビットの下書きのヘッド（この fork では `STRATA_MTP_SKETCH=0` のときだけ使います。1 ビットの写しが同じ役を担うためです）、`STRATA_PARK_VRAM_MIB`、`STRATA_ROUTE_DUMP` / `STRATA_ROUTE_REPLAY`、`STRATA_DOWN_WS`（Pascal 向け、APU の行とは組み合わせません）、READY の前にページの移動が落ち着くまで待つこと、`STRATA_SETTLE_S`、Qwen3.5-9B（密なモデル）とそのエンコーダー。
- setup と tools: Gyro-L と Gyro-Coder、Qwen3.5-9B、ウォームアップの待ちの表示、VRAM が 8 GB 未満の GPU への案内、量子化の物差しの長い文書のセット、`tools/task_check.py`。
- 共有の部品: 両方のエンジンが呼ぶオプションの解析（`parse_shared_option`）、DONE 行、エキスパートの層の入れ替えの規則、PCIe の分担の設定。generate.cpp はそれらを呼びつつ、この fork 自身の判断（下の「残したもの」）を保っています。

fork の形を残したものは次のとおりです。両側が同じ Strata の変更を別の形で持っていた所と、890M 向けの既定です。

- KVA（`STRATA_KVA`）: この fork の identity モードだけの形。上流の capture / oracle のモードは入れていません。
- 推測デコードの費用の規則（`STRATA_SPEC_COST`、HIP の統合 GPU で既定）と lookup の窓の別の表（`STRATA_SPEC_SOURCE_COST`）、`STRATA_SPEC_ADAPT_WARM`、起動の profile の充填のスレッド（`STRATA_FILL_THREADS`）、PLE の表の RAM の方針の既定、会話のキャッシュの裏のコピーの統合 GPU での既定、`STRATA_DEQUANT_SIDE` の既定、GDN の WMMA の再帰（`STRATA_GDN_CHUNKED`）、APU の down の行、サブグループの xor の交換、`STRATA_VERIFY_ECAP` と `STRATA_ROUTE_STATS`、`STRATA_RAM_ADAPT` の fork の形。
- decode の注意のカーネル（qsa_decode_attn.cpp）: 上流の 2026-10-27 のカーネルは入れましたが、KV ヘッドのセルを計算ユニットの数に合わせた枠に分け、枠ごとに複数の塊をオンラインの softmax で足す形（merge が読む部分和が減ります）は、足す順が変わるので greedy の答えを変えます（bench の 6 チャットのうち 2 つ）。890M の decode は同じ速さでした（3 周の交互で 31.6〜32.0 対 31.5〜31.9 tok/s）ので、`STRATA_QSA_CU_SLOTS=1` の opt-in にし、既定では以前どおり塊ごとに 1 枠です。スコアのループが 1 歩で読むセルの数を GPU ごとに測る部分（`STRATA_QSA_NC`）と問いのヘッド 4 つの和はそのまま入っています。merge のほかの部分は答えを変えません。
- README とこの fork の文書、`container/`、CI、リリースの確認の宛先。

確かめたこと: free ビルド、CTest（pack の要る 2 本を除いて通ります）、serve の Python の試験（1,016 本、28 本は skip）、tools の試験（388 本、8 本は skip）、`reuse lint` と `tools/license_class.py`、Python と JavaScript の lint で merge が新しく足した指摘がないこと。
Mage-Flow-Turbo 以外の画像のモデル、contrib-llvm のビルド、Windows のクロスビルド、ほかの GPU は未検証です。

## DFlash2 の drafter の移植

upstream の Strata 0.2.10 の既定の drafter、DFlash2（incoai/Qwen3.8-Flash-Next-DFlash2、Apache-2.0）を、upstream 自身の SYCL 版（`sycl/` の DF2-SYCL、Intel Arc 向け）からこの fork に移しました（2026-11-04）。`--draft dflash2 --draft-model PACK` で選ぶ任意の機能で、既定は MTP のままです。使い方は docs/DETAILS.ja.md の「DFlash2 の drafter」にあります。

- 移したもの: drafter のカーネル（Q4_0 の GEMM、2 tap の動的 conv、窓つき attention、top-16、selector の歩き。`src/kernels/xe/dflash2.cpp`）、drafter 本体（pack の読み込み、K/V の ring、ring の記録と組み直し、1〜8 行の round の graph。`src/core/dflash2.cpp`）、verify の窓と prompt の経路の tap（層 5・15・25・35・45 の完了出力、hc 4 本の平均）、generate の CLI と serve の経路、窓の規則（selector の確率の積が `--draft-conf` を下回ったところで窓を終える）、pack を作る `tools/dflash2_pack.py` とその試験、ring の記録の試験（`dflash2_ring_test`。pack があるときだけ動き、組み直した K/V の行が元の更新と同じビットかを見ます。890M で通りました）。
- この fork での違い: GEMM の既定の variant はGPU が報告する sub-group の大きさで選びます（16 がある Intel は upstream の SIMD16 の対のカーネル、ない AMD と NVIDIA の device compile は 32 lane のカーネル）。AMD の device compile ではカーネルが work-group の大きさを宣言します（レジスタの割り当て）。drafter の stream、graph、メモリは `strata/core/gpu.hpp` を通ります。
- 入れていないもの: `--draft auto` と MTP への fallback、batch slot と layer split の DFlash2、Q4_0 の selector codebook（upstream の SYCL 版も bf16 だけ）、lookup を先に取る規則の配線（`STRATA_DF2_LOOKUP_SKIP`、既定 0）、layer-major の prefill の tap、serve・setup・app の組み込み（pack の取得と変換、Drafting の選択）。pack は手で作ります。

計測（890M、abliterated IQ2_XS、bench の 6 チャット、greedy、`--spec 6`、枠 24,576、1 周ずつ）:

| drafter | decode（tok/s） | 下書きの採択 / 提案 | drafter の 1 round |
|---|---|---|---|
| MTP（sketch、CUM 0.5。既定） | 31.0 | 1,313 / 2,045 | 約 4 ms |
| DFlash2、confidence 0.35 | 29.2 | 1,333 / 2,362 | 7.9 ms |
| DFlash2、confidence 0.6（既定） | 29.5 | 1,243 / 1,707 | 7.8 ms |
| DFlash2、confidence 0.8 | 27.9 | 1,141 / 1,357 | 7.7 ms |

交互 3 周（枠 24,576、冷却つき）: MTP 31.6 / 31.7 / 32.0 tok/s、DFlash2（confidence 0.6）29.6 / 29.6 / 29.8 tok/s で、中央値で 6.7% 遅いです。drafter は 7.7 ms/round、2,191 トークンに 942 round（1 round 2.3 トークン。MTP は 874 round、2.5 トークン）。6 チャットの greedy の答えは 3 周とも MTP と同じでした。チャットごとに見ると、コードと英語の 3 つ（クイックソート 37.2 対 36.5、transformer 33.2 対 32.4、bash 42.6 対 40.5）では DFlash2 が 2〜5% 速く、日本語の文章の 2 つ（四季 21.5 対 25.6、光合成 25.0 対 28.8）で 13〜16%、TCP/UDP（32.9 対 34.0）で 3% 遅いです。

1 round に受け入れるトークンの数は MTP と同じほど（約 2.5）ですが、drafter 自身が 1 round に 7.8 ms かかり（5 層の Q4_0 の重み 0.5 GB を 890M の帯域で読む時間が大半）、MTP の約 4 ms より長いので、decode は 6〜7% 遅くなります。upstream が AMD の HIP で MTP より 4〜12% 速かったのは、帯域が 2.5〜6 倍のカード（Strix Halo、R9700）です。Issue #23（2026-10-24）の見積もり（drafter の重みの読みが MTP の 4 ms を超え、+2〜3% が上限）のとおりでした。drafter は下書きを変えるだけで、答えは verify が決めます。

upstream が Strata-rdna3 から取り込んだ attention のカーネル（d31256b79）も、opt-in で移してあります（`STRATA_DF2_ATTN=3`）。Q と K と転置した V を FP16 でローカルメモリーに置き、S = Q K^T と O += P V を RDNA の行列命令（WMMA）で計算します。float64 の参照との差は出力の RMS の 0.2〜0.4% です（`dflash2_attention_parity`）。890M の交互 3 周（6 チャット、greedy、400 トークン）で、drafter は 7.89 / 7.72 / 7.72 ms から 7.61 / 7.57 / 7.56 ms/round になり、decode は 36.23 / 36.96 / 36.92 対 36.75 / 37.01 / 37.01 tok/s でした。同じ周の MTP は 40.81 / 42.27 / 42.02 tok/s です。答えは 3 周とも MTP と同じでした。drafter の時間の大半は重みを読む時間なので、attention を速くしても MTP との差は埋まりません。既定は FP32 のカーネルのままです。

## Strata 0.1.69 の取り込み

upstream の Strata 0.2.21 から 0.2.22（エンジンの版は 0.1.69、2026-11-25）までの 39 コミット（merge を除きます。0.2.21.1 も含みます）を 1 件ずつ見ました。一覧は `.upstream-0169.md`（リポジトリの根、git の外）です。
エンジンの変更は、複数 GPU の layer split と Inkling-2-Mini の CUDA だけで、Qwen3.8-Flash-Next を 890M で動かす経路は変わっていません。

移したものは次のとおりです。

- serve: `/v1/responses` で custom tool（Codex の `apply_patch`）の引数をモデルが書いている間も、`KEEPALIVE_S` ごとに keep-alive を送ります（e83de74d7）。引数を出し終えるまでストリームに何も流れず、止まったストリームを切るクライアントやプロキシが、正常な要求を途中で終えていました。
- setup: 実行の設定を書く前に、その置き場のフォルダーを作ります（2fdbdb56d）。
- web: API キーが要るという 401 のメッセージに、キーがどこから来るか（`--api-key`、`STRATA_API_KEY`、実行の設定の `api_key`）を足しました（51644762f、284e2c5b6）。

入れていないものは次のとおりです。

- Inkling-2-Mini の 11 件（tool の結果の名前、sampling を指定しない要求へのモデルカードの sampling、思考の予算と段階、CPU で読む画像の縮小、setup の HIP の計画など）。Inkling-2-Mini の変更は取り込まないことにしています（2026-11-25）。
- `--mtp` のとき sampling を指定しない要求を greedy にする規則を、Qwen3.6 だけに限る変更（1e5070f73）。XeStrata はこの規則（d35353903）を入れていないので、Qwen3.8-Flash-Next はもとからモデルカードの sampling で動きます。
- 会話キャッシュを RAM に合わせて 8 GiB より大きくする変更（6c3f3fbb4）。大きさは空きの RAM からモデルのファイルなどを引いて決めるので、64 GB の 890M と Qwen3.8-Flash-Next では 8 GiB を越えません。
- 複数 GPU の layer split の lend の予約（7ab0d923c、2aed34d4e）、READY を待つ時間の数え方（89ef8456a。XeStrata は、エンジンが CPU やディスクを使っている間は待ち続けます）、AVX-512 の CPU での drafter の自動選択（289fa51de。XeStrata の setup は DFlash2 を自動では選びません）。

## Strata 0.1.67 の取り込み

upstream の Strata 0.2.19 から 0.2.21（エンジンの版は 0.1.67、2026-11-24）までの 293 コミット（merge を除きます）を 1 件ずつ見て、エンジン・serve・setup・tools の層に当たるものを XeStrata-890M に移しました。一覧は `.upstream-0167.md`（リポジトリの根、git の外）です。
大半は NVIDIA、Intel の Battlemage と Panther Lake、Windows、複数 GPU 向けの変更と、app と Terminal と Control の層、Hy4-Flash のエンジンで、この節で移したものには入っていません。

移したものは次のとおりです。

- エンジン: batch のスロットに入る会話は、接頭辞になる checkpoint をすべて持ちます（1f1057b3f）。会話の途中から分かれる後続の要求が、メッセージの境目から再開します。
- エンジン: 新しい ids の接頭辞になっている古いスナップショットも、会話キャッシュから落とします（f87121058）。park の stage は、ストリームの K/V では索引の行だけを数えます（ad4c565e8 の一部）。890M でストリームの K/V を使う起動では、stage が共有の RAM を 9 倍ほど多く取っていました。
- エンジン: `STRATA_WATCHDOG_S` と `STRATA_WATCHDOG_IO_S` の値を確かめ、範囲の外なら既定値に戻します（dca06a36d）。prompt と `max_new` が文脈に収まるかは、桁あふれしない形で先に確かめます（534802c8d の #6493）。GGUF の数の配列は 1,024 要素まで読みます（3319f9e27）。
- serve: engine が死んだときの 503 に、終了コードと最後の ERR の行を載せます（f1f91d97b）。10 分に 3 回死んだ engine は起動し直しません（264b155dd、6a6e56d55）。
- serve: 要求の画像は、結合ファイルに写し終えるまでキャッシュから追い出しません（91fb63284）。読み手が読めなかった画像は、PNG に直して 1 回だけ送り直します。`/v1/models` は答えの長さの上限も返します（f5ab600d5）。
- Inkling-2-Mini: SYCL の窓の K/V リングを、窓と 1 回の step の容量を足した大きさに広げました（64c224f3c）。リングが窓ちょうどの大きさだったので、窓をまたぐ prefill の塊が、同じ塊のトークンがまだ読む key を上書きしていました。収まらない `--parallel` は起動時に断ります（de1768f2b ほか）。
- Inkling-2-Mini: ggml-org 版の GGUF、drafter のテンソルの名前と型、写し終えた mmap のページの返却（a8e30004a、de3cb634a、d528772fc、8c2afbc2b ほか）。GGUF と safetensors のモデルを動かせるのは CPU の backend だけで、SYCL の backend は .strata 以外を起動時に断ります（1178074aa）。

入れていないものは次のとおりです。

- Inkling-2-Mini の新しい機能（思考の段階、slot の振り分け、drafting の調整、SYCL で GGUF を直接読むこと、診断の道具など、36 件）。この fork の目標は Qwen3.8-Flash-Next なので、取り込まないことにしました（2026-11-25）。
- CPU のエキスパートの AVX-512 化（4414c2988）。890M のふつうの起動では、エキスパート 24,576 個がすべて GPU のキャッシュに入るので、CPU のエキスパートは動きません。
- upstream の HIP の Unsloth の量子化カーネル（9e3ad2778、5ea521afe）。890M で upstream の HIP 版を測ると、v0.2.21 の decode は v0.2.19 より速くなっていました（22.0〜22.2 対 19.9 tok/s、交互に 3 周）。伸びの一部は、gfx1150 でも既定になった expert の mode 8 で、`STRATA_HIP_EXP8=0` で切ると 20.9〜21.7 tok/s でした。XeStrata は mode 8 を前から既定にしています。同じ回の XeStrata は 41.7〜42.5 tok/s でした。
- MTP のとき、sampling の値のない要求を貪欲にする変更（d35353903）。upstream の engine は sampling の要求で MTP を止めますが、XeStrata は止めません。取ると既定の要求の答えが変わります。890M では、モデルカードの sampling の decode は貪欲より約 9% 遅くなりました（37.8〜38.5 対 41.7〜42.5 tok/s、交互に 3 周）。速さを取るなら、設定の sampling に temperature 0 を書きます。
- `--batch` の複数スロット向けの MTP の融合と QSA（0f1738e61、CUDA）、鍵ごとの同時実行数（7bff4c47a）、アラビア文字の draft vocab（2a8265942）、別のサイズのモデル ID の 404（fba071562）。

## Strata 0.1.65 の取り込み

upstream の Strata 0.2.15 から 0.2.19（エンジンの版は 0.1.65、2026-11-22）までの 1,082 コミット（merge を除きます。0.2.16 までが 609、その先が 473）を 1 件ずつ見て、エンジン・serve・setup・tools の層に当たるものを XeStrata-890M に移しました。一覧は `.upstream-0162.md` と `.upstream-0165.md`（リポジトリの根、git の外）です。
大半は、NVIDIA と Windows と複数 GPU 向けの変更、app と Terminal と Control の層、新しいモデルの Hy4-Flash のエンジンで、この節で移したものには入っていません。
serve のツール呼び出しの読み手は、この fork が引数の型を schema から読むので、upstream の差分はそのままでは当たりません。当たらなかった分は fork の形に書き直しました。

移したものは次のとおりです。

- エンジン: `STRATA_ROUTE_TAIL_SKIP` の既定をオフにしました（upstream 962475995。答えの質を理由にした upstream の判断です）。`STRATA_ROUTE_TAIL_SKIP=7` でオンにできます（[DETAILS](DETAILS.ja.md)）。890M のふつうの起動では、エキスパートの 90% 以上がキャッシュに入るので、前から働いていませんでした。
- エンジン: cgroup の中での RAM の空きの読み方を直しました（cfb3d9bcb、ad9aa7597）。cgroup v1 の制御と `0::` の行が混ざる host では v1 の上限を読み、`memory.max` のない group は親の上限を読みます。読めなかったときは、何が読めなかったかを誤りの文に添えます。
- エンジン: PLE の表を読んでいる間も、serve が停止を見張るための進行を数えます（c369e7392、fbd35fc21、fdba9ebd9）。遅いドライブでの読みを、止まったエンジンと取り違えません。
- エンジン: `STRATA_ROUTE_PRIOR`（opt-in）が、縮退したトークンで番兵の値を ids に書かないようにしました（709d0a99c、5144e8b53）。オンの間は、READY の前の窓の下準備も飛ばします（027aac9bd）。
- エンジン: CPU のエキスパートの窓が、GPU が書き換える ids を、処理の始めに一度だけ写して読みます（f0f39de26）。
- エンジン: 会話キャッシュは、入りきらない会話を、何かを追い出す前に断ります（4c3470aa8）。追い出してもメモリーが戻らないときは追い出しを止めます（96edb9337、a1b7cd186）。
- エンジン: serve で prompt の読み込みが GPU の誤りで失敗したら、すぐに終わります（85c949f85 の考えを fork の形で）。
- エンジン: `--batch-mtp` のスロットごとの draft の数を BDONE の行で返します（5152c0366）。f32 から bf16 への変換は NaN を NaN のまま保ちます（fda204b5c）。`--max-context` が学習時の文脈を超えていて rope scaling がないときは WARNING を出します（252e6ea0c）。
- serve: ツール呼び出しの読み。崩れた閉じタグや開きタグ、`<arguments>` の JSON、Claude の `<invoke>` の形、思考の中の `<function=NAME>` だけの呼び出し、文の中の対にならないバッククォートなどを読めるようにしました（8426249f0、05795a84a、7c743ad7f、db0c38425、859f0da47、c447a2850、20056dbbe、5959daece ほか）。ストリームで流す長い引数の解析は線形にしました（fb3c3ea69）。integer や number の引数に書かれた `+10`、`010`、`.5` を数として読みます（bee824a22）。
- serve: ツールを宣言した要求の返答が、動かさない呼び出しの形で終わったら、同じ要求の中で 1 回だけ書き直させます（e05e6e0cc、`"tool_markup_retry": false` か `STRATA_TOOL_MARKUP_RETRY=0` で切れます）。
- serve: 返し方。200 を返したあとの誤りは、その API の最後のフレームで伝えます（cf0743747）。想定外の誤りには、接続を切らずに 500 を返します（e626616e4）。要求の欄の型が違うときは、欄の名前を言う 400 を返します（c2767a851、84ae5707b、3755e94a3）。サーバーが止めたことで切れた要求は 503 にします（b1cdaa389）。
- serve: `/v1/models` が文脈の上限を `max_model_len`、`context_length`、`max_context_length` でも返し、`GET /models/{id}` にも答えます（dd6da3a9d、18cce81d0、23d94d00a）。思考の上限は `thinking_token_budget` と `thinking_budget` からも読みます（7ffd66628）。
- serve: `--prompt-cache-every` の定期チェックポイントを、再開した位置からの格子に置きます（2c8d300ad）。temperature 0 の要求には、モデルカードのペナルティを付けません（d7ce708ba、3313dbd61）。server の glibc のメモリー区画を 2 つに抑えます（aeecf0f5d の server の側）。
- serve: Responses API の `max_output_tokens` で切れた答え、Codex の multi-agent v2 の item、クライアント側の `tool_search`（3af02433a、3cb933d29、277af44e9、e04cb092f）。
- エンジン: DFlash2 の drafter の attention を RDNA の行列命令で計算するカーネルを、opt-in で移しました（`STRATA_DF2_ATTN=3`、upstream d31256b79。数字は上の DFlash2 の節にあります）。
- 新しいモデル: Thinking Machines の Inkling-2-Mini を、upstream の専用のエンジン（`strata-inkling-2-mini`）ごと移しました（[DETAILS](DETAILS.ja.md) の説明は、2026-11-27 にこのエンジンと一緒に外しました）。upstream の CUDA・HIP・Metal のバックエンドは取らず、SYCL のバックエンドを AMD の 32 レーンの sub-group で動くように直しています。モデルの名前は略さずに書く決まりなので、upstream の `ink_` などの名前は `inkling2mini_` に置き換えました（`tools/upstream/inkling2mini_names.sed`）。
- setup: RAM の予算（`--resident-budget-gib`）の規則を、RAM から 17 GB を引く形にしました（8780e9fdc、c8d70bbcd）。GPU が RAM を共有する PC は、upstream と同じく 24 GB を引く元の規則のままなので、890M では変わりません。
- setup と tools: Coder のモデルの pin（3d54c7137）、`mtp_fetch.py` の `--repo`（c9b6956c6）、`--batch` のスロットを上書きされた会話の続きを確かめる試験の道具（6da257d80）ほか。

入れていないものは次のとおりです。

- MTP なしのセッションの保存と復元の修正（1b58c335d）。この fork は MTP なしのセッションファイルを断るので、同じ不具合は起きません。
- systemd-oomd からモデルの読み込みを守る pacer（02c4bd423、aa754707c、cc0242cc8）。
- prefill のエキスパートを FP16 の行列命令で計算するカーネル（`STRATA_PF_F16=2`、upstream bbb22af24 と 7c3f971e1。upstream も gfx1150 では既定にしていません）。upstream の HIP 版そのものを 890M で動かして比べました。3,419 トークンの prefill で、エキスパートの積にかかる時間は upstream の既定で 4,995 ms、`STRATA_PF_F16=2` で 3,638 ms、この fork の int8 のカーネルで 3,719 ms（gate/up 2,530 + down 1,189）です。fork のカーネルと差がないので、移していません。prefill 全体では、upstream は `STRATA_PF_F16=2` で 3〜5.5% 速くなり（16,085 トークンで 114〜117 → 121〜123 tok/s、交互 3 周）、fork は同じ長さで約 316 tok/s です。
- 新しいモデルの Hy4-Flash（Tencent、220B）のエンジン。公開されているいちばん小さいファイル（SQ-64）が 62 GB で、ほかに 67.6 GB の Engram のファイルが要ります。RAM 62 GiB の 890M の PC では動かして確かめられないので、移していません。
- cgroup v1 の page cache の加算（c9f031dcd）と、`--prefill` の塊の上限（c983ecc64）。前者が前提とする upstream の部品が、この fork にありません。後者は CUDA の grid の行数に合わせた修正です。

## Strata 0.1.61 の取り込み

upstream の Strata 0.2.13.1 から 0.2.15（エンジンの版は 0.1.61、2026-11-16）までの 736 コミット（merge を除き、0.2.14 と 0.2.14.1 を含みます）を 1 件ずつ見て、エンジン・serve・setup・画像の読み手の層に当たるものを XeStrata-890M に移しました。一覧は `.upstream-0161.md`（リポジトリの根、git の外）です。
大半（570 件）は CUDA と gfx12 の prompt のカーネル、Windows の Strix Halo、DeepSeek V4.1 を外す作業、app と Terminal と Control の層のもので、この fork には当たりません。
統合 Radeon の長い prompt を upstream の int8 のカーネルで読む変更（69394994a）は、この fork が前から自前の SYCL のカーネルで読んでいるので取っていません。

移したものは次のとおりです。

- エンジンと serve: thinking budget の上限で、エンジンが思考をちょうどそのトークンで止めます（GEN の `hold=N:ID`、READY の `hold` と `holdb`、DONE の `hold`。upstream be101072b、5aa7f1d88、b92075d92、02020478b）。締めのパスは、エンジンがいま持っている状態から続けます。890M で上限 24 トークンの要求を送ると、思考は 24 トークンで止まり、締めのパスは 104 トークンのうち 85 トークンを読み直さずに続けました（62 + 24 − 1）。
- エンジン: server が STOP を送ったあとに終わった decode の窓を、確定せずに捨てます（upstream 33648fef1、`STRATA_SERVE_STOP_SKIP=0` で切れます）。この fork の GDN の書き戻しを遅らせる設定（`STRATA_GDN_ACC`）で溜まった行は、捨てる窓の前に書き戻します（この fork の手当て、`Verifier::drop_window`）。
- エンジン: 追い出した会話のメモリーを C ライブラリーのヒープに残さず、OS に返します（upstream 8c71c4927）。
- エンジン: READY の前に、推測デコードの窓の幅を 1 回ずつ回してグラフを先に作ります（upstream 84e11a773、`STRATA_WINDOW_WARM=0` で切れます）。この fork の方策は 1 から S までのどの幅も使うので、すべての幅を回します。ある幅が失敗しても、残りの幅は回します。
- エンジン: 置いた会話の K/V のバッファーを取る領域（segment pool。upstream e04c4c931、b0c818ad7、6aba6d5fb、8f4a2e0af、07cf3f4af、01e21ef20、ef371eba8、`STRATA_PARK_SEGPOOL_MIB`）。upstream のもう 1 つの領域、checkpoint pool はこの fork にないので、取っていません。RAM を確かめて置いた会話を追い出すときは、追い出すたびに領域の空いている分を OS に返します（upstream は最初の 1 回だけで、追い出した会話の分が領域に残り、RAM が増えないまま会話をすべて追い出すことがありました）。
- エンジン: Linux の HIP の統合 GPU で、KFD がキューを止める操作を要求の最中に起こしにくくする手当て（upstream 0a17fb99e、8f6393a37、0ee263992、6d8e42886、f9932c6ae、6069c9d0c、f73fa7410、03e687ace、f55afa152）。upstream は gfx11 の統合 GPU の名前で選んでいましたが、この fork では KFD の topology とデバイスの報告で選びます（[DETAILS](DETAILS.ja.md#amd-の-apu)）。59375eb06 は、この fork に前からあります。
- serve: パイプが閉じていても STOP で例外を出しません（bbf97633c）。boolean の引数に文字列で書かれた true と false を真偽値として読みます（1aabfb1ce）。待ち受けの bind で getfqdn を引きません（3e1fcaaf6）。子のプロセスに `GGML_NO_BACKTRACE=1` を渡します（38ad0bcbd、`STRATA_BACKTRACE=1` で渡しません）。画像の読み手の READY を 1 つの期限で待ち、ほかの行は記録に残します（4cce1ee4d）。止めているあいだに来た要求には 503（`server_draining`）を返します（3cf22f5ef）。
- setup: 低 RAM の判定で GiB と GB を比べていた単位のずれを直しました（9a779fb40）。並列ダウンロードの最初の range の問い合わせを再試行します（95e9b7341）。コンテナの memory の上限が RAM より小さければ、それを RAM として数えます（81ac7c8ef）。ダウンロードの試験を安定させました（f328c2cd2）。
- 画像の読み手: CPU 版の `strata-vision` の自前の読み手（`tools/vision/own`、upstream 75ee0ea87 ほか 14 コミット。[DETAILS](DETAILS.ja.md#画像)）。890M の既定の Vulkan 版は mtmd のままです。

890M の 6 つのチャット（greedy、200 トークン、枠 24,576）を、この取り込みのビルドの既定と、KFD の手当て・窓の下準備・segment pool を切った形で交互に 3 周回しました。
答えは 3 周とも同じで、decode は 32.3 / 33.7 / 33.0 対 32.8 / 33.3 / 32.7 tok/s でした。最初の要求は窓の下準備で 37.1〜39.7 から 40.2〜41.4 tok/s になり、起動から最初の答えまでは KFD の待ちの分だけ 2〜5 秒長くなります。

入れていないものは次のとおりです。

- 780M（gfx1103）だけの FP16 の GEMM の既定（8d7f997d4）と、統合 Radeon の int8 の prompt のカーネル（69394994a、9187ef658）。
- checkpoint pool（`STRATA_CKPT_POOL`）。これが前提とする upstream の ca378670b ほかの変更が、この fork にありません。
- split engine の hold の検査（5aa7f1d88 の一部）。この fork の split（`serve/split.py`）は `strata --serve` を 2 つ使い、decode の側のエンジンが hold を受けます。
- 画像の読み手の GPU の塔（`vit_gpu.cu`、`vit_metal.mm`）の変更と、古い CPU 向けのビルドの下限（`STRATA_ISA_FLOOR`）。この fork の CPU 版は AVX2 を下限とする portable のビルドだけです。

## Strata 0.1.59 の取り込み（Qwen 4 の形状）

upstream の Strata 0.2.12 から 0.2.13.1（エンジンの版は 0.1.59、2026-11-09）までの 219 コミットを調べ、Qwen 4 系の形状に対応する作業の最初の部分だけを入れました。一覧は `.upstream-0159.md`（リポジトリの根、git の外）です。版は xe0.2.12-1 のままです。
この fork は、upstream が 2026-10-12 から持つ geometry（モデルの形状を GGUF から読み、カーネルのコンパイルした形状と照らす仕組み）を取り込んでおらず、形状は `core/layout.hpp` と `artifact/model_spec.hpp` で読んでいました。fork の SYCL のカーネルは `src/kernels/xe/` にある独自の移植なので、upstream の `sycl/` の差分は `patch` では当たらず、行ごとの置き換えで入れました。

入れたものは次のとおりです。

- コンパイルするモデルの形状をヘッダで選びます。`-DSTRATA_GEOMETRY=<名前>`（`flash_next`（既定）、`q4_lite`、`q4_mid`、`q4_big`、`q4_deep`）か、`-DSTRATA_GEOMETRY_FILE=<パス>`（まだ木にない形状のヘッダ）で指定します。指定しなければ、これまでのビルドと同じです。形状は `include/strata/core/geometry.hpp` の `ModelGeometry` と `kCompiledGeometry` で、どの形状も満たすべき条件は `geometry_checks.hpp` の `static_assert` が検査します。`geometry_from_spec` は、GGUF の値を既定値ではなくこの形状と比べます。
- `prefill.cpp`、prefill のカーネル、`fused_gr`、`s2_expert_grouped`、`native_ple_postops`、`qsa_prompt_attn`、`ngram`、CPU のエキスパートの定数は、`kCompiledGeometry` から取ります。Flash-Next の値は変わりません。890M の IQ2_XS で、設置済みのエンジン（xe0.2.10-1）と 6 チャットの greedy の答えが同じで、slots も同じ 24,576、decode は 36.03 対 36.05 tok/s でした。
- `tools/q4geo_synth.py`（任意の形状の合成 GGUF と、その形状のヘッダ）、`tools/qwen4_intake.py`（GGUF のヘッダを読み、新しいモデルを動かすのに何が要るかを言う道具）、`strata_tokenizer.pre_tokenizer_kind`、`serve/toolformat.py` を入れました。

`q4_lite`、`q4_mid`、`q4_big`、`q4_deep` のどの形状のビルドでも、エンジン全体（`strata` と試験）がビルドでき、その形状の合成モデルが 890M のエンジンで動きます。形状ごとの分岐は足さず、head の数や幅は `kCompiledGeometry` から取ります。

- hyper-connection の読み（`fused_gr`）は、hc_lr 256〜512（64 刻み）と n_embd 4096 までの形状で動きます。Q8_0 の塊の幅は n_embd から、up の行の段の数は hc_lr から取ります。1 トークンの読みは、正規化した行がデバイスのローカルメモリーに収まらなければ、複数トークンの読みに回します。`gr_bench --ref`（CTest の `gr_read_ref`）は、読みを double で計算した CPU の参照と比べます。`q4_lite`、`q4_mid`、`q4_big` のどの経路も、差は rms の 1e-6 未満でした。Flash-Next の読みは、変更の前とビットまで同じです。
- prefill の GDN の kernel は、コンパイルした形状の head の数（ssm_k_heads と ssm_v_heads）で実体化します。head の数に既定値はなく、呼ぶ側が渡します。plain のモデル（head の数を GGUF から読む経路。key head 16、value head 32）の kernel は、コンパイルした形状と head の数が違うビルドだけが追加で持ちます。plain のモデルの head の数がどちらとも違えば、読み込みの時点で断り、`--prompt-chunk 0` を案内します。matrix-engine の scan（`STRATA_GDN_TC`）は、key head 1 つあたり 4 つの value head（64）まで受けます。
- CTest の `gdn_prompt_parity` は、prefill の GDN の gates、conv、recurrence を、double で計算した CPU の参照と比べます。recurrence は、FP32 の kernel、chunked、scan の 3 つの経路を比べ、出力の gate は sigmoid と SiLU の両方で比べます。890M では、head が 16×32、16×48、16×64 のどれでも、FP32 の kernel と scan の差が rms の 1e-4 未満、FP16 を使う chunked の差が rms の 2% 未満でした。decode の 1 トークンの kernel は、`gdn_parity --heads K,V` で head の数を変えて比べられます。
- decode の attention の kernel は、コンパイルした形状の head の数で実体化します。890M では、prefill の attention もこの kernel です。`qsa_dense_attn_parity` は、コンパイルした形状でも CPU の参照と比べます。
- Flash-Next の幅でデータを作っていた試験（`expert_layout_test`、`file_expert_source_test`、`q2_bitplane_parity`、`native_expert_parity --synthetic`、`native_dense_ple_key_test`）は、コンパイルした形状の幅でデータを作ります。`expert_layout_test` のうち、`tests/data` にある Flash-Next の pack を読む部分は、幅の違うビルドでは理由を出して飛ばします。
- `--mtp` と `--draft dflash2` は Flash-Next の drafter を読むので、幅の違う形状のビルドでは起動時に断ります。
- エンジンがエキスパートのキャッシュの大きさを決める構成（`--expert-cache auto`）で、prefill が自分のバッファを持つときは、そのバッファのためのメモリーを残します。残す量は、Flash-Next の幅のビルドでは upstream の式（160 MiB と、1 トークンあたり 680 KiB）のままです。ほかの形状のビルドでは、prefill が確保するバッファを数えた値（`Prefill::bytes_needed`）です。upstream の式は、2,048 トークン以上の chunk が取る ring（いちばん大きい blob の数百個ぶん）を含みません。890M で chunk が 2,048 トークンのとき、数えた値は q4_lite が 1,578 MiB、q4_mid が 3,818 MiB、q4_big が 5,106 MiB です。Flash-Next の IQ2_XS は 1,383 MiB で、式の 1,520 MiB に収まります。これらは、エキスパートの行列積をすべて int8 で計算する実行の値です。エンジンは起動時に int8 と FP16 の速さを測って、どちらで計算するかを決めます。FP16 も使うと決まった実行では、数えた値が q4_big で 230 MiB ほど、q4_mid で 160 MiB ほど増えます。

合成モデルは、`tools/q4geo_synth.py --small-ple` の GGUF、`tools/iq_pack.py --compat-bf16` の pack、`tools/make_profile.py --no-base` の expert profile で作ります。q4_lite、q4_mid、q4_big は 4 層にしました（ヘッダは `-DSTRATA_GEOMETRY_FILE` で渡します）。q4_deep は、形状どおりの 64 層、512 エキスパートで作りました（98 GiB）。890M で、299 トークンを prefill で読み、84 トークンを greedy で生成した結果は次のとおりです。

| 形状 | 合成モデル | prefill | decode（`--spec` 2、4、6） |
| --- | --- | --- | --- |
| q4_lite | 4 層 | 2,671 tok/s | 178、156、141 tok/s |
| q4_mid | 4 層 | 1,128 tok/s | 79.3、60.9、53.9 tok/s |
| q4_big | 4 層 | 559 tok/s | 54.4、49.1、42.0 tok/s |
| q4_deep | 64 層、512 エキスパート | 2.9 tok/s | 0.92、1.03、1.00 tok/s |

q4_deep の合成モデルは、エキスパートが RAM に入りきりません。ファイルから読みながら動くので、この速さになります。

q4_deep の合成モデルでは、エンジンが「verify: timed out」で止まることがありました（16 回のうち 2 回）。host USM に置いたエキスパートのキャッシュ（38 GiB）が swap に出ていたためです。キャッシュのページが swap に出ると、ページが戻るまで、ドライバーがそのプロセスの GPU のキューを止めます。エキスパートの重みは圧縮が効かないので、RAM の中の swap（zram）に出してもメモリーは空かず、swap がいっぱいになるまで回収が続きます。止まった実行では、直前の 18 秒間に、キャッシュの 5 GiB ほどが swap に出ていました。
エンジンは、host USM のキャッシュを `mlock` で RAM に固定するようにしました（[DETAILS](DETAILS.ja.md#ram-がモデルより少ないとき省-ram-モード)）。固定した実行は、14 回とも最後まで動きました。decode の途中でカーネルに 14 GiB の回収を頼むと、固定しない実行は timeout で止まり、固定した実行は最後まで動きました。Flash-Next の設定も、このキャッシュを使います。固定の前と後で 6 チャットの greedy の答えは同じで、decode は、3 回ずつ交互に測って、前が 35.5〜36.4 tok/s、後が 36.1〜36.5 tok/s でした。

重みは乱数なので、答えの中身は確かめられません。確かめたのは、推測デコードの窓（`--spec` 2、4、6）を変えても、2 回実行しても、greedy の出力が同じことです。Flash-Next は、変更の前と 6 チャットの greedy の答えが同じで、decode は、3 回ずつ交互に測って、変更の前が 35.0〜36.5 tok/s、後が 36.4〜36.5 tok/s でした。

入れていないものは、新しい形状のモデルを feed で配る仕組み（`strata_control` と setup の shape engine）、CUDA と HIP の geometry、Metal、Windows の Strix Halo の変更です。この fork にその層がなく、SYCL のカーネルに同じ切り替えもありません。SYCL の Q5_K と Q6_K の down は、fork の `native_expert_supported` が前から受けています。

## Strata 0.1.58 の取り込み

upstream の Strata 0.2.11 から 0.2.12（エンジンの版は 0.1.58、2026-11-07）までの 84 コミット（merge を除く）のうち、エンジン・serve・setup の層に当たるものを XeStrata-890M に移しました。前回と同じく Strata のコミットを `patch -F3` で当て、ずれた hunk は手で置き直しました。一覧は `.upstream-0158.md`（リポジトリの根、git の外）です。
0.2.12 の upstream の変更の多くは、Windows の Strix Halo（gfx1151）の CUDA/HIP のカーネルと既定の表、gfx12、NVIDIA、CPU のエキスパートの AVX-512 の行、peer GPU、Metal、app と Terminal と Control のもので、この fork には当たりません。Strix Halo の変更は 890M と同じ RDNA3.5 向けですが、UD-Q4_K_XL と UD-IQ4_XS-v31 の K-quant と Q5_0 のエキスパートの .cu のカーネルで、この fork の SYCL のカーネルには同じ切り替えがありません。

移したものは次のとおりです。

- エンジン: verify の flag B を copy stream に積んだコピーの後ろで上げます（upstream PR #5096）。空の計画の層が、前の層のコピーが終わる前に flag B を上げてしまう不具合を直しました。この窓でコピーを積んでいないときは、これまでどおりその場で上げます。adaptive tier の 1 回分のコピーの範囲は、pipelined の回をまるごと指すようにしました（PR #4844、adaptive_range.hpp と adaptive_range_test）。
- エンジン: PLE の読み手の issue() は 64 行以下の要求を stack の配列でまとめ、hash table と job の vector を作りません（890M の decode は PLE の行をファイルから読みます）。
- エンジン: `STRATA_PREFILL_TIMING` は同じ段階の印を続けて記録しません（PR #4006）。合計は同じで、event の記録が減ります。
- エンジン: Strix Halo の prefill の変更のうち、down の行を FP16 で書くもの（6e31946cb）は、考え方だけを移しました。int8 の down の積（iq_mmq）が行を FP16 で書き、combine がそれを読むので、1 層が書いて読み戻す行の量が半分になります。HIP の統合 GPU では既定で入り（`STRATA_PF_DM16=0` で止まります）、ほかの GPU では `=1` で入ります。CPU の分担や helper GPU のある層は FP32 のままです。890M の IQ2_XS で、15,251 トークンの prompt の prefill は 313.3 tok/s から 316.5 tok/s（+1.0%）になりました。FP16 の丸めのぶん、答えは変わります。
- エンジン: `STRATA_DF2_GATE=1`（opt-in）は DFlash2 の窓を要求ごとに見積もった下書きの生き残りから決めます。window_rule_test を CTest に入れました。
- serve: ツール呼び出しの読み方を直しました。`<function=tools NAME>` と `<function=tools>`（#5409）、綴りの違う opener の `<toolcall>` と `<tool-call>`（#5276）、MCP のツールを名前の最後の段で書いた呼び出し（#5281）を、宣言したツールの呼び出しとして読みます。tool_choice が required でツールが 2 つ以上なら、`<tool_call>` だけを書きます。宣言していないツールの呼び出しを返す設定 `undeclared_tool_calls`（`STRATA_UNDECLARED_TOOL_CALLS`、既定は text）を足しました（#4622）。これらの検査は buffer の末尾だけを見ます。
- serve: 要求の計数。譲った（BYIELD）prompt の読みの時間とトークンを 1 回だけ足し、admission の時間を二重に数えず、エンジンが報告しなかった数は 0 ではなく不明（None）のままにします。
- setup: SHA-256 が合わないダウンロードは消さずに `<name>.bad` として残します。次の実行でもう一度 hash を確かめ、合えばそれを使います。新しいダウンロードが合うか、空きが足りなければ消します。
- 試験: Anthropic の tool_result を tool_use_id の順に並べる試験（並べる処理はこの fork に前からあります）、Anthropic の route がどのモデル名でも読み込んだモデルで答える試験。

入れていないものは次のとおりです。

- 統合 GPU で起動時の空きメモリーが少ないときの文（530a29ea2）。upstream は Strix Halo の resident RAM モードのときだけ「--vram-reserve-mib を足せ」と言わなくなります。890M では案内のとおりに予約を増やすと数字が上がるので、この文は残します。890M の host USM のエキスパートのキャッシュは、空いている RAM から予約を引いた分だけを取るからです。ほかのプログラムが RAM を 13 GiB 使っている状態で、予約が 1,624 MiB のときの空きは 0 MiB、案内どおりの 3,760 MiB のときは 1,948 MiB でした。
- 統合 GPU で RAM の予算がエキスパートのキャッシュに残らないときの文（1d342f083）。Windows の Strix Halo ではキャッシュをグラフィックスのメモリーから取りますが、890M の Linux では GTT がシステムの RAM なので当たりません。
- ファイル層の Linux の既定を全エキスパート常駐のとき止める変更（994029b02）と、ファイル層の再検査（0a0e4463e）。890M の構成はこの経路を通りません。
- CPU のエキスパートの AVX-512 の行と道具、STRATA_MANIFEST、STRATA_PF_ALLOC_TRACE、token forcing の道具（force_check.py）。

decode の計算は変えていません。prefill の計算は、HIP の統合 GPU で down の行が FP16 になったところだけが変わります。CTest は 167 件中 165 件が通り、落ちた 2 件（expert_parity、pool_test）は Strata 形式の pack がこの PC にないためのものです。serve の試験 1,173 件と tools の試験 458 件が通ります。

## Strata 0.1.57 の取り込み

upstream の Strata 0.2.10 から 0.2.11（エンジンの版は 0.1.57、2026-11-05）までの 205 コミットのうち、エンジン・serve・setup の層に当たるものを XeStrata-890M に移しました。前回と同じく Strata のコミットを `patch -F3` で当て、ずれた hunk は手で置き直しました。一覧は `.upstream-0157.md`（リポジトリの根、git の外）です。
0.2.11 の upstream の変更の大半は、NVIDIA のカードの既定（K-quant の MMQ、32768 トークンの塊、FP16 の重みのキャッシュ、CPU と PCIe の分担）、Linux の gfx12（R9700）の既定、複数のカードの split エンジンと layer split、Windows の Strix Halo、Metal、app と Terminal と Control のもので、この fork には当たりません。

移したものは次のとおりです。

- setup: Unsloth が 2026-11-04 に出し直した Qwen3.8-Flash-Next の 7 セット（UD-IQ1_M、UD-Q2_0、UD-IQ2_XS、UD-Q2_K_XL、UD-IQ3_XXS、UD-Q3_K_XL、UD-IQ4_XS-v31。revision d9620119、SHA-256 と大きさの pin、読めない UD-IQ1_S は断ります）をカタログに入れました。この fork では experimental の印が付き、890M では動かしていません。統合 GPU でエキスパートが RAM に収まるセットは RAM の予算を取らず（uma_holds_experts）、890M ではそのあとに host USM の設定（xe0.2.10-2 の uma_host_cache）が書かれます。
- エンジン: その 7 セットの gate/up と down の型の組（IQ1_M・IQ2_XXS・IQ2_XS・IQ2_S・Q2_0・IQ3_XXS・IQ4_XS と Q2_0・IQ4_NL・Q5_0・Q8_0 の 28 組）の native_expert_parity を CTest に入れ、890M で 28 組とも通りました。合成の重みを量子化するとき、importance matrix の要る型（IQ2_XXS、IQ2_XS、IQ1_M）には平らな行列を渡します（渡さないと ggml が止まります）。Q5_0 の down の GPU の許容は 1.8e-2 に締めました。upstream は Q5_0 の内積の -16 の項を、量子化した活性化の和で取るように直しましたが、この fork は d652cd6 を移したときから同じ形です。
- setup: pin のある shard は hash を確かめてから完了の印を付けます（upstream #4088）。`quant_formats --hf` はファイルの大きさを tree の一覧から取ります（#5075）。tools/strata_bench.py は鍵を語として伏せます（#4484）。tools/stream_ownership_probe.py を入れました。
- serve: 同じモデルを 2 つ動かす replicas で、要求は自分の pipe を交換の終わりまで持ちます（#2892。答えが別の要求に混ざる不具合）。`--batch` では frame と交換の終わりが 1 つの要求と 1 つのエンジンの process に属し（#4451）、vram() と session_file() も control の行を要求と同じ順で取ります。読まなくなった streaming の client はエンジンの turn を持ちません（#4271、`STRATA_STREAM_WRITE_TIMEOUT_S`、既定 60 秒）。console の出力は有限の queue を通ります（#4479、`STRATA_BLOCKING_CONSOLE=1` で元に）。control の行を待つ要求が wake-up を落とさず（#4738）、起動に失敗したエンジンの process を別の thread が閉じても落ちず（#4792）、断った batch の admission は error の行になり、エンジンの出力の UTF-8 でないバイトも読みます（#5044）。終わりの event なしに終わった要求は、同じ process で誰も turn を持たないときだけエンジンを終え、「次の要求で起動し直す」と言うのは本当に終えたときだけです（#3961）。ConvCacheLog.poll は lock の下で動きます（#3443）。長い会話の分岐の follow-up は、その会話の slot に行きます（#2518）。閉じタグを並べた tool の引数は、閉じタグが届く間は保留します（#5040）。1024 未満のポートを断る文は、Linux と macOS の理由を言います。
- serve の安全: Pillow を 13.0.0 にし（GHSA-78g4-xqwv-q3g8、JPEG 2000 のデコーダーのメモリの書き込みの不具合）、開く画像の形式を JPEG、PNG、GIF、BMP、WEBP、TIFF、AVIF に限ります。requirements.txt も同じ pin です。rpm と Podman は Fedora の python3-pillow を使います。
- web: `/?q=` は質問を入れて focus し、Send を押すまで送りません（#4221）。

入れていないものは次のとおりです。

- Strix Halo で 16384 トークン超の塊を起動時に断る probe。HIP の runtime の API で書かれていて、890M の setup は塊 2048 を書き、エンジンは MemAvailable からキャッシュの大きさを決めます。
- low-RAM モードの写し先のディスクの空きの検査。この fork の low-RAM は RAM に写し、packs フォルダーへは写しません。
- `--batch` の slot ごとの DFlash2 の下書き、名前で選んだ DFlash2 が要求の数を保つ規則、layer split の起動の retry。この fork にその経路がありません。
- gfx12 の prompt の staging ring と、NVIDIA と gfx12 の既定。この fork の prefill は自身の WMMA の経路です。

エンジンの decode と prefill の経路は変えていません（エンジンの変更は parity 試験だけです）。CTest は 164 件中 162 件が通り、落ちた 2 件（expert_parity、pool_test）は Strata 形式の pack がこの PC にないためのものです。serve の試験 1,113 件と tools の試験 454 件が通ります。

## Strata 0.1.56 の取り込み

upstream の Strata 0.2.9 から 0.2.10（エンジンの版は 0.1.56、2026-11-02）までの 708 コミットのうち、エンジン・serve・setup の層に当たるものを XeStrata-890M に移しています。上流の XeStrata（MistVVK/XeStrata）の main は 2026-10-30 で止まっていて 0.2.10 を取り込んでいないので、前回と同じく Strata のコミットを `patch -F3` で当て、ずれた hunk は手で置き直しました。一覧は `.upstream-0156.md`（リポジトリの根、git の外）です。
この間の upstream の変更の大半は、DFlash2 の drafter（99 コミット）、Linux で複数のカードに 1 つのモデルを割る split エンジン、DeepSeek の ISTA-DAS Coder、Mac の DeepSeek の画像、8 GB のディスクリート GPU での 128K、gfx12 と CUDA のカーネルの既定、Metal、Windows、app と Terminal と Control のもので、この fork には当たりません。

移したものは次のとおりです。

- エンジン: repeat・frequency・presence のペナルティは、答えが書いたトークンだけを数えます。プロンプトの中の数字や引用した前の答えがそれだけで罰せられることはなくなり、`STRATA_PENALTY_PROMPT=1` で前の窓（プロンプトと答えの直近 `penalty_last_n` トークン）に戻ります（upstream #4590、penalty_window_test）。
- エンジン: 要求は出せる行だけを commit します。`--max-tokens` や EOS で止まる要求の、出せない行がキャッシュに残らず、保存した会話が客が受け取ったものと一致します（upstream #4677。upstream は HIP だけで既定にしましたが、この fork は全バックエンドで既定です。`STRATA_SERVE_STOP_BOUND=0` で切ります）。
- エンジン: 致命の GPU のエラー（メモリアクセスの違反、デバイスの消失）のあとは、エンジンが `ERR fatal GPU error: ...` を出して自分で終わり、サーバーは要求を 503 で返して次の要求でエンジンを起動し直します。10 分に 3 回目で起動し直しを止めます（upstream #4350）。upstream は CUDA の sticky error を読みますが、SYCL にはないので、失敗した呼び出しが返したエラーの文で見ます。`STRATA_INJECT_FATAL=N` は試験の仕掛けです。
- エンジン: PLE の表を RAM に固定する前に memlock の soft limit を hard limit まで上げ、足りないときは固定できる量と設定する値を言います（upstream #4671）。エキスパートの arena の RAM が待っても足りないときは、この固定を返して RAM を空けます（upstream #4846）。
- エンジン: pack が BF16 のまま持つ射影（Heretic v2 IQ4_NL の層 0 の attn_gate）を、ほかの層と同じ Q8_0 にして出します（upstream #4741）。仮想メモリのない runtime で `--vram-elastic` を指定しても、キャッシュを 1 ブロックにして起動します。
- serve: `/v1/messages` の Anthropic のツール検索（`defer_loading` のツール、`tool_reference`）に対応し、宣言にない名前の呼び出しを `tool_use` として返します。会話で既に呼んだツールは Anthropic と Responses の経路でも宣言済みとして扱い、`tools.write`・`functions.write`・`default_api.write` の形の名前は `write` として読みます。呼び出しの閉じタグが値を終えるのは、あとに本文がなく閉じタグが続くときだけです（ファイルの中に閉じタグを並べた値が切れなくなります）。
- serve: `/v1/messages` の context overflow は Anthropic と同じ文で始まります。読んだあとのファイルを閉じて `ResourceWarning` をなくしました。
- setup: ダウンロードがサーバーに届かないときに proxy の設定の仕方を言います。読み取り専用のモデルフォルダーの完了の印は Strata 自身のフォルダーに置きます。
- web: 速さのカードは、要求がないときは run の平均を出します。

入れていないものは次のとおりです。

- DFlash2 の drafter（upstream の 0.2.10 の既定の drafter）は、この取り込みのあとに別の作業で移しました（次の「DFlash2 の drafter の移植」）。
- 12 行と 16 行の prompt lookup の窓（`--lookup-max-t`）。この fork に移して 890M で測りましたが、速くならなかったので入れていません（2026-11-06）。ファイルの書き直し 3 課題（greedy、交互 2 周）は 8 行 47.4 / 45.6、12 行 46.4 / 44.8、16 行 31.8 / 47.1 トークン/秒で、16 行の 1 周目は VRAM が空いていない中で窓の graph を作った分だけ遅くなりました。6 チャットでは長い窓に出した下書きがほとんど通りませんでした。8 行を超える窓は kernel の数が 2 倍になり、HIP の kernel 引数のバッファーに収まらず、launch と host の待ちが互いを待って止まるので、移すなら launch を別のスレッドで行う必要があります。
- RAM が足りないときに arena を `--mmap-experts` の形に落とす一連の変更。890M の setup は初めから `--mmap-experts` を選びます。
- ready file の直後の stop の扱い（この fork の serve に ready file はありません）と、Qwen の on-demand の画像の読み手（この fork の serve には、それが前提にする elastic な読み手の仕組みがありません）。

decode の計算は変わらないので、6 チャットの greedy の答えが設置済みのエンジンと同じことだけを確かめました。

## Strata 0.1.55 の取り込み

upstream の Strata 0.2.8 から 0.2.9（エンジンの版は 0.1.55、2026-10-31）までの 169 コミットのうち、エンジン・serve・setup の層に当たるものを XeStrata-890M に移しています（Issue #76）。上流の XeStrata（MistVVK/XeStrata）はまだ 0.2.9 を取り込んでいないので、Strata のコミットを 3-way で当て、ずれは手で置き直しました。
この間の upstream の変更の大半は、Metal、DeepSeek と Qwen3.6 の CUDA のエンジン、CUDA の tensor core と gfx12 の GDN、`--batch` の co-read と研究の切り替え、replica、Windows、app と Terminal、upstream 自身の sycl/ のもので、この fork には当たりません。

移したものは次のとおりです。

- serve: 会話で既に呼んだツールは宣言済みとして扱い、その呼び出しを文でなくツール呼び出しとして返します。コードフェンスの中と字下げしたコードブロックの中に書かれた呼び出しは実行しません。
- setup: 子の出力に UTF-8 でないバイトがあっても落ちません（`errors="replace"`）。
- prefill: `STRATA_PREFILL_TIMING=1` が hyper-connection の読みを down+silu、up、inject、mix に分けて出します（15K トークンの prompt では 10.6% が 0.1 + 3.1 + 2.9 + 0.7 + 3.4%）。

まだ入れていないもの: CUDA の hyper-connection の読み v3（5〜8 トークンの窓で warp あたりの行を増やす。fork の fused_gr.cpp で sub-group あたりの行を増やす形になりますが、890M の窓は 3〜4 行なので当たる窓は少ないです）。
エンジンの計算は変わらないので、取り込み前後の速さの比較はしていません。

## Strata 0.1.54 の取り込み

upstream の Strata 0.2.7.1 から 0.2.8（エンジンの版は 0.1.54）までの 70 コミットのうち、エンジン・serve・setup の層に当たるものを XeStrata-890M に移しています（Issue #56）。
上流の XeStrata（MistVVK/XeStrata）が 2026-10-30 に SYCL の形で取り込んだ 13 コミットを 3-way で当てました。
この間の upstream の変更の大半は、upstream 自身の sycl/ の移植（Battlemage の prompt の注意、2 枚の Arc の流れ作業）、DeepSeek、app と Terminal、Mac のもので、この fork には当たりません。

移したものは次のとおりです。

- エンジン: エキスパートの領域に要る RAM を最大 20 秒待ってから読み込みます。方針で RAM に固定した PLE の表が自動の会話のキャッシュの余地を奪うときは、固定を返します。バッチの空いた枠の退避で、本体の写しと共有の根のチェックポイントを消しません。AVX-512 のない CPU で止まるときは、原因が CPU でなく `native_experts.txt` のないパックだと言います。
- serve: ツール呼び出しの回復（`tool_call_recovery`）を既定で入れ、`"tool_call_recovery": false` で切ります。壊れた引数、余分な思考の閉じタグ、`</function>` の抜けで呼び出しを失わず、呼び出しを丸ごと引く値や同じ引数名が続く値を値の一部として読みます。
- setup: 使わない APU に大きな BIOS の割り当てがあるときは、OS が失う RAM と直し方を言います。

移していないのは次のとおりです。

- upstream の sycl/ の変更（lane-cell の prompt の注意、問いのまとめ、2 枚の Battlemage の流れ作業の既定）: この fork のエンジンは別のコードです。
- 検証とバッチの窓を 16 行にする HIP の既定: `--batch` の 1 組 8 枠を超えるときだけ効きます。
- 自動のエキスパートキャッシュが読み込みの後の確保を見込む変更: この fork の設定は `--expert-cache` を明示し、XeStrata にはその警告がありません。
- route prior の既定: この fork は 0.2.7 の自動の 1.5 を取っていないので、upstream が既定から外した理由を文書に書いただけです。
- 終わり印を 1 トークンのものに限る変更: fork の `stop_token_ids` がすでにそうしています。前の run が残したエンジンの待ちは `strata_control/procs.py` に依るので、エンジンの側の待ちだけを入れました。
- DeepSeek、app と Terminal と Control、Mac、DGX Spark、2 枚の GPU の既定。

確かめたのは、free のビルド、CTest（pack のない 2 本を除いて通ります）、serve の Python の試験（852 本、12 skip）、890M での decode の前後の比較（abliterated 版 IQ2_XS、greedy、交互 3 周: 前 31.42・31.47・31.43 tok/s、後 31.07・31.41・30.38 tok/s、答えは同じ）です。
contrib-llvm のビルド、Windows のクロスビルド、ほかの GPU は unverified です。

### 実測で差が出なかった変更の見直し（Issue #57）

2026-10-31 からの方針として、890M の実測で差が揺れの中だったことだけを理由に、変更を不採用や opt-in のままにしません。答えが同じで、はっきり遅くならない限り、論理的に速くなる変更は既定に入れます。
この方針で、`STRATA_SPEC_COST` を HIP の統合 GPU の既定にし（上の推測デコードの項）、重みの 16 bit の読みに AMD の device compile だけ non-temporal の hint を付け（重みは窓ごとに 1 回しか読まないので、キャッシュを活性化と格子表に残します）、prefill の MMQ の 16 行のカーネルで重みの生のワードを 1 ステップ先に読むようにしました（カーネル単体で IQ2_XS の 16 行が 594 → 575 us、ビット同一）。
密な GEMV の 2 行 / サブグループを 5 列以上で使う規則は、今の既定の IQ4_XS の LDS 版と同等で 8 列の ssm_out では 10% 遅いので、入れていません。
890M の decode（abliterated 版 IQ2_XS、greedy、交互 3 周）は取り込み後 31.88・29.40・28.47 tok/s、見直し後 31.90・30.79・31.71 tok/s で答えは同じ、15K トークンの prompt の読みは 314.3・313.8・311.3 → 314.9・314.4・314.5 tok/s で同じです。

### 上流 XeStrata の opt-in の速さの移植（Issue #58）

上流の XeStrata が Strata 0.2.7 の取り込みで SYCL の形で持っていた opt-in の速さの変更を、Issue #57 の方針で移しました。
会話のキャッシュの裏のコピー（`STRATA_CKPT_ASYNC`・`STRATA_PARK_ASYNC`・`STRATA_CKPT_GATHER`）は統合 GPU の `--serve` で既定に、密な重みの逆量子化を別のストリームで重ねる `STRATA_DEQUANT_SIDE` は HIP の統合 GPU の既定にしました（15K トークンの prompt の読み 314.1・305.8・310.8 → 311.1・308.1・308.6 tok/s、答えは同じ）。
swiglu と int8 の量子化を 1 つにする `STRATA_SWIGLU_QUANT` は 890M で 15K トークンの prompt が 11% 遅く、層ごとの歩き方でエキスパートのページを温める `STRATA_PREFILL_WARM` は 890M ではエキスパートが RAM にあるので、どちらも opt-in のままです（[DETAILS](DETAILS.ja.md)）。
890M の decode（abliterated 版 IQ2_XS、greedy、交互 3 周）は移す前 31.52・30.87・31.66 tok/s、移したあとの既定で 31.62・30.73・30.41 tok/s で、答えは同じです。

### layer-major prefill と塊の大きさ、attention の work-group の上限（Issue #59、#63）

上流 XeStrata が SYCL に移した layer-major prefill（`STRATA_PREFILL_LAYER_MAJOR=1`。大きな塊を層ごとに読み、エキスパートの転送を塊の数ぶんの 1 にし、ルーティングされた行を群ごとの積にまとめる）は入れていません。890M では 15K トークンの prompt の GPU 時間のうち転送に当たる段階（gather）が 0.9% でコピー待ちは 0 なので、転送を減らしても得はありません。得があるのはエキスパートの積が 1 回に持つ行の数で、同じ効果は `--prefill 8192` で出ます（1 周目 283〜294 → 324 tok/s）。しかし 62 GiB の RAM では 2 周目に swap に出て 71 tok/s に落ち、layer-major も同じ大きさの残差の行と群のバッファを置くうえ、和の順が変わって答えも変わるので、塊は 2,048 のままです（[DETAILS](DETAILS.ja.md#amd-の-apu)）。
MarchRaBBiT/XeStrata の、RDNA の行列の attention に work-group の上限を渡してレジスタのあふれを減らす変更（dd9c2a66a）は、この fork では 2026-10-14 の `STRATA_AMD_REQD_WG` が同じ上限を付けていて、gfx1150 の WmmaKernel<256,12,4> は VGPR 236 であふれ 0、PromptAttnMma は 216〜236 であふれ 0 です。足しても ISA は同じなので、変えるものはありません。

### ほかの fork の serve の直し（Issue #66）

Ben1078/Strata の 8b90ab807 を移しました。`"effort_position": "end"` のとき、要求の最後の `<system-reminder>` だけの user の発言を答えの直前の system のターンに移し、次のターンが会話のチェックポイントを再利用します（[DETAILS](DETAILS.ja.md#api)）。
shoftee/Strata の 2feece509 も移しました。停止のときは listening socket を先に閉じ、エンジンには QUIT の前に STOP を書いて要求を終えさせてから、handler のスレッドを待ちます（shoftee の計測で、忙しい停止が 20 秒 → 約 1 秒）。閉じた pipe への書き込み（ValueError）もエンジンの終了として扱います。
Ben1078 の e8e0cdc40（GGUF のメタデータだけの shard）は、この fork の gguf_reader.hpp に既にありました。

### 統合 GPU のメモリーの決め方と expert source の上限の検査（Issue #67）

shoftee/Strata の 3 つを移しました。expert cache の大きさは cgroup の上限で丸めた値でなく `MemAvailable` から決め（48defc12e。cache のページは cgroup に課金されない。ただし host USM に置く cache はプロセスのページで課金されるので、2026-11-19 から cgroup の上限の中で決めています）、統合 GPU の `--serve` で `--conversation-cache-mib N` を書いたときは自動の expert cache の reserve に N を足し（ff5809230）、ArenaExpertSource の prefetch・release・pinned に層の上限の検査を足しました（f6094ff1e。172K〜190K の prompt で表の外を読んで止まっていた）。fatal signal の backtrace は入れていません。
Ben1078 の bd0e37bb4（HIP の expert の arena の mlock）は入れていません。mmap のエキスパートはページキャッシュで、mlock の対象ではありません。host USM に置く expert cache はプロセスのページで、ドライバーは固定せず、swap に出ることがありました。そのため、あとからエンジンがこのキャッシュを自分で `mlock` で固定するようにしました（[Strata 0.1.59 の取り込み](#strata-0159-の取り込みqwen-4-の形状)）。

### lookup の窓と MTP の窓で別の費用の表、O_DIRECT の開き直し（Issue #64）

htbmixbox/Strata-890M の ba96856d1 を移しました。推測デコードの DraftPolicy は、lookup（prompt-lookup、suffix）の窓と MTP の窓の 1 周の時間を 1 つの EMA で持っていましたが、lookup の窓は下書きの層を走らせないので同じ幅でも費用が違い、互いの周で値段を付けていました。`STRATA_SPEC_SOURCE_COST` で lookup の窓の時間を別の表（文脈の bucket ごと）に持ち、lookup を選ぶ側はその表を見ます。測っていない幅は共有の見積もりのままで、gated の経路（defer_cost / apply_costs）も共有の表のままです。HIP の統合 GPU では既定で on、`STRATA_SPEC_SOURCE_COST=0` で 1 つの表に戻ります。
890M の decode（abliterated 版 IQ2_XS、greedy、交互 3 周）は移す前 31.53・30.38・26.66 tok/s、移したあとの既定で 30.93・30.02・31.78 tok/s、`STRATA_SPEC_SOURCE_COST=0` で 28.86・30.14・28.37 tok/s で、答えは同じです。要求ごとの値に熱の落ち込みがあり、周ごとの差は揺れの中です。
同じ fork の d3ed6bb4a も移しました。`O_DIRECT` を断るファイルシステムでは DirectFile がページキャッシュを通して開き直します（[DETAILS](DETAILS.ja.md#困ったとき)）。

### 全エキスパートが cache にあるときの file tier、streamed walk の下限、decode の時間の内訳（Issue #69、#71、#72）

01xjw/Strata の 735a40ec3 を移しました。cache を満たしたあと全エキスパートが GPU の cache にあれば、Linux の既定で入った io prefetch（と decode gate）を切り、routing-aware prefetch（層ごとに router の積を CPU で計算する）を始めません（[DETAILS](DETAILS.ja.md#amd-の-apu)）。890M の decode（abliterated 版 IQ2_XS、greedy、交互 3 周）は前 31.65・31.71・31.65 tok/s、あと 32.35・32.29・30.64 tok/s（3 周目は 1 つの要求が熱で落ちた）で、答えは同じです。同じ Issue の STRATA_MMAP_PREFETCH（CPU の pool が計算するエキスパートの先読み）は、この設定では CPU がエキスパートを計算しないので入れていません。起動の profile の充填の並列化（chenyb999 の STRATA_FILL_THREADS）は起動の時間の話で、まだ測っていません。
Indigoshores/Strata（Strix Halo）の考えのうち、streamed walk を 16 トークンから使う形は `STRATA_STREAM_ALL_MIN` で試せるようにしました（既定は 2048 のまま）。523 トークンの prompt では既定と 16 がどちらも 1,844 ms で同じで、得はありません。3 段の高さのタイルは、この fork の MMQ が行の数でカーネルを選ぶ形で既にあります。PLE の先読みは、この fork が表を RAM に固定するので当たりません。
decode の時間の内訳（`STRATA_DECODE_TIMING=1`、6 チャット）は、1 窓 74〜87 ms のうち GPU の待ち 61〜71 ms、下書き 5〜7 ms、層ごとの host 0.2 ms 未満です。decode は 1 窓 1 グラフなので、HIP のカーネルを 1 つずつ起動する前提の persistent な起動（unimatrix099、ryan4yin）は当たりません。

### 起動の profile の充填を複数のスレッドで（Issue #74）

chenyb999-zhcn/Strata の 0c16171ce を移しました。pair の admit は順に 1 スレッドで行い、blob の読みと cache へのコピーは互いに重ならない slot を 8 スレッド（`STRATA_FILL_THREADS`、0 か 1 で 1 スレッド）で行います（[DETAILS](DETAILS.ja.md)）。890M（abliterated 版 IQ2_XS、24,576 のエキスパート、約 35 GB）では充填が 28.0・25.9・15.7 秒 → 13.2・13.4・12.5 秒、ready まで 52・63・42 秒 → 42・43・37 秒で、decode と答えは同じです。コピーは SYCL の queue の同期の memcpy のままなので、8 スレッドでも 2.6〜2.8 GB/s にとどまります。host USM の cache なら CPU の memcpy で書ける（起動の充填では GPU がまだその slot に触っていない）のが次の候補です。

### 起動の充填を CPU の memcpy で（Issue #75）

host USM の cache の slot を起動の充填で CPU の memcpy で書く形を `STRATA_FILL_HOST_COPY=1` で試せるようにしました（opt-in）。890M では SYCL の queue のコピーと同じ 12.4 秒（2.85 GB/s）で、ready までも 43・37・38 秒対 43・36・37 秒と同じです。上限は mmap のファイルの読みで、コピーではありません。答えは同じです。

## Strata 0.1.53 の取り込み

upstream の Strata 0.2.3.1 から 0.2.7.1（エンジンの版は 0.1.53）までの 796 コミットのうち、エンジン・serve・setup・tools の層に当たる 596 コミットを見て、33 コミットの変更を XeStrata-890M に移しています（Issue #49）。
この間の upstream の変更の大半は DeepSeek のエンジン、Metal、DGX Spark、NVIDIA のカーネルのもので、この fork には当たりません。

移したものの主なものは次のとおりです。

- serve: 宣言していない名前のツールの呼び出しを文のまま返すとき、その名前を `strata_rejected_tool_calls` で答えに添え、`/metrics/prometheus` で数えます。名前空間つきのツールを `namespace=member`、`namespace_member`、`member` だけの綴りと、`__` を `_` に潰した綴りでも読みます。
  `--host ::` で IPv6 で待ち受け、`/v1/responses` は `max_output_tokens` の 0 と -1 を上限なしと読んで答えに `timings` を付け、`repair_latex_escapes` で LaTeX のエスケープで壊れた JSON を直してから確かめます。
  Chat Completions のツールの結果を呼び出しの順に並べ直し、message のない要求は 400 にし、画像のエンコーダーが起動できなければ 503 を返します。
- エンジン: n-gram の表を、RAM に表とエキスパートの置き場と 8 GiB の余りが入るときは RAM に置きます（`STRATA_PLE_AUTO_RAM=0` で止まります。890M の 62 GiB では置きません）。
  PLE の直接読みで、失敗した読みが次のチケットを永久に待たせていたのを直し、表を複数のスレッドから使うときの保護を足しました。
  `STRATA_RAM_ADAPT` は層ごとに 4 組まで入れ替え、stage のバッファーにあるエキスパートはファイルを読まずに写します。`--adapt-decay` は 0〜1 だけを受けます。復元した会話の retained K/V を、別の会話を置くときに使いません。
- setup と tools: cmake と ninja は venv のものを先に取り、古い cmake は使いません。別の generator で configure された build フォルダーは作り直します。
  ダウンロードは、長さが取れなければ公開された大きさを使い、2 分データが来なければそう言います。pip の記録はどの OS が書いたかを覚えます。gguf_reader はヘッダーの読みに上限を置き、calibrate は計測の途中でエンジンが落ちても続けます。

移していないのは次のとおりです。

- DeepSeek のエンジン（src/dsv41、trellis）、Metal と Mac、DGX Spark（arm64 の CUDA ビルドと sm_121 の表）、NVIDIA のカーネルと既定（sm_80・sm_89・sm_120 の表、MMVQ-IL、fused の prompt、qsa_select）、2 枚の GPU の layer-major prefill と pipelined batch、gfx12（R9700）だけの既定、Windows の gfx1151 だけの既定。
- upstream の製品層（app、Terminal、Control、packaging、docker）に依る setup と serve の変更。
- 会話キャッシュの CUDA の copy stream に依る変更（HIP に copy stream がない）、checkpoint の写しの失敗を sticky error で見分ける変更、KVA の capture と oracle のモード、upstream のブラウザでの試験（tools/webtest）。
  QFUSE の commit の修正と q8_1 の除算の修正は、fork の窓が self-commit し、除算がどれも IEEE なので当たりません。

取り込みのときに見送り、2026-10-29 に Issue #50〜#55 で移したもの: DGX Spark 向けの setup の値（`--mtp-hnorm stream` を統合 GPU の既定に）、KVA（`STRATA_KVA`）、SPEC_PROB の提案則と窓に依らない CPU の行（#3416）、AVX-512 の gather の行と exact rows、会話の退避とチェックポイントの失敗の扱いと窓のグラフの解放、web ページの改装（履歴、多言語、KaTeX、プレビュー、Load / Unload、MCP の絵）。
移植のあとに確かめたのは、free のビルド、CTest（pack のない 2 本を除いて通ります）、serve の Python の試験、890M での decode の前後の比較（abliterated 版 IQ2_XS、greedy、交互 3 周: 取り込み前 31.91・31.88・31.65 tok/s、取り込み後 31.90・31.88・31.97 tok/s、移植後 30.45・31.91・31.95 tok/s、答えは同じ）です。既定の設定では速さも答えも変わらず、速くなるのは setup の既定に入れた `--mtp-hnorm stream`（+5%）と opt-in の `STRATA_KVA=1`（15K トークンの prompt の読みが +18%）です。

確かめたのは、free のビルド、CTest（pack のない 2 本を除いて通ります）、serve と tools の Python の試験、890M での decode の前後の比較（abliterated 版 IQ2_XS、greedy、交互 3 周: 前 30.93・31.46 tok/s、後 31.79・31.46・31.44 tok/s、答えは同じ）です。
contrib-llvm のビルド、Windows のクロスビルド、ほかの GPU は unverified です。

## Strata 0.1.48.1 の取り込み

upstream の Strata 0.1.45 から 0.2.3.1（`7ddc17a3`、エンジンの版は 0.1.48.1）までの 1,116 コミットのうち、エンジン・serve・setup・tools の変更を XeStrata-890M に移しています。
upstream の 0.2 は app と Terminal と Control の層を加えた版で、エンジンはその中の 1 つの部分です。

移したものの主なものは次のとおりです。

- 推測デコード: 費用を見て MTP の窓を決める `STRATA_SPEC_COST`。下書きごとの確率から幅ごとの期待トークン数を求め、測った採択率と 1 周の時間で割って比べます。
  890M の 6 チャット（abliterated 版、28 W、`--spec 6`、greedy、3 回の交互の計測、2026-10-23）の 3 周の中央値は既定（`STRATA_SPEC_CUM`）30.9 tok/s、`STRATA_SPEC_COST=1` 31.5 tok/s で差は揺れの幅の中でしたが、費用の表で負ける幅の下書きをやめる規則なので、2026-10-31 に HIP の統合 GPU の既定にしました（Issue #57。`STRATA_SPEC_COST=0` で `STRATA_SPEC_CUM` の規則に戻ります。ほかの GPU では opt-in のままです）。
  推測の輪の始まりの規則を `spec_loop_runs()` に名づけ、実行の始めだけ適応の段の入れ替えを頻繁にする `STRATA_SPEC_ADAPT_EVERY` と `STRATA_SPEC_ADAPT_WARM`（opt-in）を足しました。
  計測用の `STRATA_SPEC_TRACE`、`STRATA_SPEC_T_CYCLE`、`STRATA_ROUTE_STATS`、`STRATA_SPEC_WATCH`、`STRATA_BATCH_TELEMETRY` は既定で無効です。
- serve: prompt の末尾を読む窓で logits が有限でなくなったら、次の要求は prompt を全部バッチの経路で読みます（#3465）。
  prompt は入るのに prompt + max_tokens が文脈に入らない要求は、400 を返さず max_tokens を残りに縮めます（既定。`--no-fit-max-tokens` で元の 400）。
  会話キャッシュの大きさを空き RAM から決めます（#3493）。API key なしでの公開は警告だけにし、`/v1/v1/messages` を本物の経路として読みます。
  tok/s の定義を 1 つにし、Anthropic の usage に cache_creation_input_tokens を返し、llama-server の thinking_budget_tokens も読みます。
  tool call の開きの揺れと別名、wrapper の区切りを直し（#3441、#3075、PR #3556）、画像のトークン数の上限に既定を持たせ、
  読み取りの起動に失敗したときの文に stderr の末尾と終了の状態を添えます。エンジンを起動できないときは 503 を返し、RLIMIT_RTTIME が有限なら警告します（#3238）。web ページの 4 つの修正も入っています。
- CPU のプール: 既定の worker 数を物理コア、host のソケット、cgroup の CPU の割り当て、測った上限（32）から決めます（pool_plan.hpp）。
  `--pool-cores`（`STRATA_POOL_CORES`、設定の "pool_cores"）で worker を置く論理プロセッサを指定でき、起動時に置き場所を 1 行で言います。エンジンの常駐スレッドに名前が付きます。
- setup と tools: モデルのファイルを 16 本の接続で分けて取り、setup が取る全部のファイルの SHA-256 と大きさを固定し、固定した版が Hugging Face から消えていたら止まります（`STRATA_ALLOW_UNPINNED=1` が逃げ道）。
  SHA-256 の合わないファイルは片ごとに直し、`--gguf-dir` の利用者自身のファイルは消しません。iq_pack は途中で止まっても次の層から続け、SSD でも RAM に余りがあれば `--ple-io ram` にし、MTP の取得は進み具合を出して再試行します。

移していないのは次のとおりです。

- upstream の新しい製品層: app（apps/）、Terminal と Control（strata_cli/、strata_control/）、packaging、Docker、integrations と、その試験と文書。XeStrata は自分の setup と serve と deb/rpm を持ちます。
  それらに依存する serve と setup の変更（TLS の証明書の確認、user_args、エンジンのパッケージ、lifeline、cloudflared）も移していません。
- 複数 GPU の選び方（gpu_roles、内蔵 GPU を最後の段にしない、hip_auto_devices）: XeStrata の serve はカードを自分の番号で選び、この機械は GPU が 1 つです。
- Gyro（APR の trellis 形式 TQ2_T・TQK6・TQK7 と Hadamard の回転）: CPU と GPU のカーネルと pack の移植が大きく、この fork の目的（IQ2_XS の decode）には効きません。
- DFlash2 の drafter（CUDA のカーネルと重み。upstream も auto では MTP のままです）。
- Linux の file tier（zero-copy、io prefetch、drive probe、prefill の planner）と NUMA のプールの分割: この機械ではエキスパートが全部 RAM にあり、ソケットも 1 つなので測れません。
- NVIDIA の世代ごとのカーネル（Volta、Pascal、sm_120、sm_86 の既定）、NVFP4、Metal、upstream の sycl/、Windows の packaging。
- HIP 固有で XeStrata にない形のもの: hipBLASLt の表、HIP の仮想メモリ、hipBLAS の warm-up、gfx12 と内蔵 GPU の WMMA の既定（fork では自前のカーネルが既定）、8 ビットのハイパーコネクション（fork の `STRATA_HC_Q8` で済んでいます）。
  queue を決まった順で開く変更は未計測です。

確かめたのは、free のビルドと、変えた C++ の試験（spec_start、draft_policy、suffix_drafter、pool_plan、pool_cores、conversation_memory、batch_telemetry）、serve と tools の Python の試験、画像の読み取り（strata-vision）の CPU のビルドです。
contrib-llvm のビルド、Windows のクロスビルド、ほかの GPU は unverified です。

## Strata 0.2.20 の取り込み

upstream の Strata v0.2.8 から v0.2.20（`a34979fc94f3d126b9bb8210f4ad311345688b52`）までの変更を取り込みました。

- DFlash2 の下書きと MTP を起動時に測って選びます。会話のキャッシュと長い文脈の処理、setup と serve も更新しました。
- XeStrata は Edge0 の latent MoE と NVFP4・MXFP4、VRAM の小さいカードで重みを流す道、画像 LoRA の層ごとの重みを足しました。D/U の量子化形式は利用できる枠から選びます。
- serve: ツール呼び出しの閉じタグ・コードの囲み・名前空間と Anthropic のツール検索を扱います。要求の停止やエラー後の処理、画像の読み手が使えない場合の CPU への切り替えも直しました。
- setup と道具: ダウンロードの範囲の検査を再試行し、固定したファイルをハッシュで確かめます。パックの保存先の空き容量を調べ、GGUF の整数テンソルも読みます。
- エンジン: 会話のキャッシュの退避で空いた RAM を返し、エキスパートの共有領域と PLE の固定を直しました。バッチのサンプリング、下書きの窓と長い文脈にも対応を足しました。
- XeStrata の共通部品にはトレリス符号の K/V、固定小数点の積、層ごとの VRAM 予算と流す道を加えました。DeltaNet の塊の走査は `XESTRATA_GDN_CHUNKED=1` で選べます。
- upstream のアプリ・ターミナル・Mac・Metal は、XeStrata の対象でないため移していません。upstream の CUDA と HIP の専用カーネルは、共通の SYCL の部品を使うためそのまま移していません。

Edge0-120B-NVFP4 の prefill / decode（tok/s）です。小さい構成は VRAM の上限を設定しています。「前」は一時領域の修正前の測定です。RX の 8192 MiB 相当は枠を前後で揃えた serve の測定、ほかは CLI です。
条件が合うときは、CLI では下書きの頭を decode の直前に用意し、それまでの領域を prompt の一時領域に使います。serve では prefill の間だけ頭の物理領域を外して戻し、prompt の塊を大きくします。

| GPU | 構成 | 1K: D/U・prefill / decode | 32K: D/U・prefill / decode |
| --- | --- | --- | --- |
| B70 | 常駐 | Q8_0・944.66 / 54.97 | Q8_0・1525.93 / 56.85（前） |
| B70 | 9700 MiB | Q4_0・593.93 / 31.94 | Q4_0・641.76 / 27.51 |
| B70 | 7700 MiB | Q4_0・618.95 / 15.36（前） | Q4_0・608.19 / 15.92（前） |
| RTX 4070 | 常駐 | Q4_0・239.26 / 46.36 | Q4_0・1181.88 / 46.93（前） |
| RTX 4070 | 8192 MiB | Q4_0・215.98 / 35.21 | Q4_0・219.31 / 20.86 |
| RTX 4070 | 6144 MiB | Q4_0・200.77 / 6.48（前） | Q4_0・119.46 / 6.88（前） |
| RX 9060 XT | 常駐 | Q8_0・520.39 / 38.39（前） | Q8_0・914.24 / 42.56（前） |
| RX 9060 XT | 8192 MiB 相当 | Q4_0・432.5 / 29.1 | Q4_0・436.6 / 26.7 |
| RX 9060 XT | 6144 MiB 相当 | Q4_0・186.00 / 18.74（前） | Q4_0・182.43 / 19.38（前） |

常駐の Q8_0 と小さい構成の Q4_0 は D/U の形式が違うので、答えも違います。

B70 の整数積の尺度を最近接に丸める修正で、答えの md5 が次のように変わりました。KL は答えが分かれるまでの行の平均 / 最大です。

| モデル・文脈 | 前 → 後（md5 先頭） | KL 平均 / 最大 | 比較した行 |
| --- | --- | --- | --- |
| Ornith-1.5 1K | bb6bcf6a → 0d11bf5f | 0.00274 / 0.00813 | 10 |
| Ornith-1.5 32K | 05bae7b1 → 2d2230e0 | 0.0362 / 0.0708 | 3 |
| Qwen3.6-35B-A3B 1K | d10297b3 → 098361d3 | 0.0645 / 0.0645 | 1 |
| Qwen3.6-35B-A3B 32K | da7576c6 → 57c0f759 | 0.00827 / 0.0729 | 25 |
| llm-jp-4.1-32b-a3b-thinking 1K | ab799775 → 42eb1b00 | 0.00473 / 0.00965 | 7 |

## Strata 0.2.8 の取り込み

upstream の Strata 0.2.7 から 0.2.8（`870aedd9`）までの変更を Xe に移しています。
この間の upstream の変更の多くは、アプリ・ターミナル・Mac・DGX Spark のものと DeepSeek V4.1-Flash への対応（後の作業に回しました）で、移していません。

移したものは次のとおりです。

- route prior（`XESTRATA_ROUTE_PRIOR`）を既定で入れない。0.2.7 ではキャッシュに載る枠が 20% 未満の GPU 1 枚で自分で 1.5 にしていましたが、upstream が長いエージェントの作業で同じ文を繰り返し結果も落ちるとして既定を戻したので、XeStrata も opt-in に戻しました（`XESTRATA_ROUTE_PRIOR=1.5` で入る）。キャッシュの小さい GPU では decode が前の版より遅くなります。context 262144 の通常使用では RTX 4070 で 2〜3 割、RX 9060 XT（キャッシュは 16%）で 1〜2 割です。キャッシュの小さい GPU では、route prior より文章を崩しにくい `XESTRATA_ROUTE_TAIL_SKIP=4`（opt-in、[詳しい説明](DETAILS.ja.md)）で、通常使用の出力が順位 7 より 13〜42% 速くなります。
- serve のツール呼び出し: 壊れた引数、余分な思考の閉じタグ、`</function>` の抜けで呼び出しを失わない。呼び出しを丸ごと引く値や同じ引数名が続く値を、値の一部として読む。この直し（`tool_call_recovery`）を既定で入れ、`"tool_call_recovery": false` で切る。
- 会話のキャッシュ: バッチの空いた枠の退避で、本体の写しと共有の根のチェックポイントを消さない。
- エンジン: エキスパートの領域に要る RAM を最大 20 秒待ってから読み込む。PLE の表を RAM に固定する方針（`XESTRATA_PLE_AUTO_RAM=1`）で、その固定が会話のキャッシュの余地を奪うときは固定を返す。AVX-512 のない CPU で止まるときは、原因が CPU でなく `native_experts.txt` のないパックだと言う。
- setup: 使わない APU に大きな BIOS の割り当てがあるときは、OS が失う RAM と直し方を言う。

upstream v0.2.8 と 3 社の GPU で比べました（IQ3_XXS の generate の 1K と 32K、serve の 32K、通常使用）。
B70 は upstream の `sycl/` 版より、generate の prefill が 2.1〜2.9 倍、decode が 35〜54% 速いです。serve の prompt は 2.1 倍、decode は 65〜66% 速いです。通常使用では、最初のトークンまで、prompt、出力のどれも速いです。
RTX 4070 は generate の prefill が 17〜74%、decode が 38〜40% 速いです。serve の prompt は 74〜75%、decode は 12〜25% 速く、通常使用もどれも速いです。
RX 9060 XT は generate の prefill が 2.1〜2.7 倍速く、decode は 0〜5% 速いです。serve の prompt は 2.5 倍速く、decode は greedy で 18% 速いです。温度 0.7 では 7% 遅いですが、これは XeStrata が書く答えが長いためで、同じ文をなぞる generate の 32K では並びます。通常使用では、最初のトークンまでと prompt は速く、出力は 1〜7% 遅いです。

移していないのは、DeepSeek V4.1-Flash への対応、アプリとターミナル、Mac、DGX Spark、upstream の `sycl/` の項目です。
次のものも移していません。

- 検証とバッチの窓を最大 16 行にする HIP の既定: 層の分割で 1 組 8 枠を超えるバッチにしか効かず、保留しました。
- prompt の注意の lane-cell と問いのまとめ: XeStrata の prompt の注意は自前の行列のカーネルです。
- 自動のエキスパートキャッシュが読み込みの後の確保を見込む変更: XeStrata にはその警告がなく、キャッシュが小さくなるだけです。
- setup が 2 枚の Xe2 に `--spec 2` を勧める変更。

## Strata 0.2.7 の取り込み

upstream の Strata 0.2.6 から 0.2.7（`5e8a4cca`）までの変更を Xe に移しています。
この間の upstream の変更の多くは、DeepSeek V4.1-Flash への対応（後の作業に回しました）、アプリ・ターミナル・Docker・Mac のもので、移していません。

移したものは次のとおりです。

- 会話のキャッシュ: 残しておいた K/V を、それを読んだ会話にしか使わない。空いたバッチの枠が decode した会話を、受け入れで上書きする前に退避する。チェックポイントの写しがメモリ不足のときは、そのチェックポイントを飛ばして要求を続ける。
- `--serve` で GPU とのリンクが 12 GB/s に満たないときは、会話のキャッシュの写しを裏のスレッドと VRAM の段に移す（`XESTRATA_CKPT_ASYNC`・`XESTRATA_CKPT_GATHER`・`XESTRATA_PARK_ASYNC`。upstream では opt-in）。RTX 4070（6.4 GB/s）では続きの要求の待ちが 226 → 201 ms になり、B70（28 GB/s）では遅くなるので入れません。
- route prior: `XESTRATA_ROUTE_PRIOR=1.5` で入る（opt-in、答えが変わる）。upstream は 0.2.7 で 13 GiB 以下の NVIDIA のカードの既定にし、XeStrata もキャッシュに載る枠が 20% 未満の GPU 1 枚で自分で入れていましたが、長いエージェントの作業で同じ文を繰り返し、結果も落ちるため、0.2.8 で upstream とともに既定から外しました。serve の decode は RTX 4070（15%）で 6〜77%、VRAM を 12 GiB に絞った B70（11%）で 15〜80% 速く、RX 9060 XT（25%）では 2〜18% 遅くなります。
- serve: OpenAI の `logprobs`・`top_logprobs`・`allowed_token_ids`、`/v1/responses` の timings と `max_output_tokens` の 0・−1、IPv6 で待ち受ける、ツールの結果を呼び出しの順に並べる、宣言のない名前の呼び出しを名指しする、Codex の名前空間の綴り、構造化出力の LaTeX の回復（opt-in）、画像のエンコーダーが起きないときの 503、MCP のツールの回数の上限で最後に 1 回答える、MCP のツールの画像（opt-in）、split serving（prefill と decode を別の GPU で、opt-in）。
- Web ページ: 会話の履歴の作り直し（保存・編集と巻き戻し・枝分かれ・答えている間の送信）、9 言語の訳（既定はブラウザーの言語）、KaTeX の数式（opt-in、`third_party/main/katex/`）、svg・html のプレビュー、読み込み・解放のボタン、広い画面、待ち行列の Timeout の印。
- setup と道具: `--rope-scale` の検査、ダウンロードの長さが取れないときの公開の大きさ、ワーカー数と画像のエンコーダーのスレッドの上限、calibrate で途中で死んだ候補だけを外す、`gguf_reader.py` の読みの上限、Maxwell 以前の NVIDIA の GPU に使えない理由を言う。
- エンジンの修正: PLE の読みの失敗の後に止まらない、prefill の領域が取れないときに部分ごとに確保しに行かない、`--adapt-decay` の範囲、`XESTRATA_TRACE` の時刻、decode の時間の行の内訳。
- opt-in のまま移したもの（XeStrata の 3 社の GPU で速くならなかったもの）: PLE の表を RAM に置く判定（`XESTRATA_PLE_AUTO_RAM=1`。B70 では 32K が完走しなくなった）、layer-major の prefill（`XESTRATA_PREFILL_LAYER_MAJOR=1`）、KVA（`XESTRATA_KVA=1`、答えが変わる、[KVA](KVA.ja.md)）、流れ作業のバッチの組の `XESTRATA_PIPED_MTP`・`XESTRATA_PIPE_ADAPT_EVERY`、`XESTRATA_RAM_ADAPT` の改良、GP100 の `XESTRATA_DOWN_WS`。
- 起動時の短い測りで選ぶもの（`0`・`1` で固定できる）: PLE の行の上げ方（`XESTRATA_PLE_COMPACT`・`XESTRATA_PLE_COPY_STREAM`。最初の起動で 512 トークンの塊を 2 通りの上げ方で測り、3% 以上速い方を選びます）。
- 2 枚の NVIDIA の `--batch` は、流れ作業の組に戻しました（直列の組は `XESTRATA_BATCH_GROUPS_NV_SERIAL=1`）。

upstream v0.2.7 と 3 社の GPU で比べました（B70 は全部の型の generate と IQ3_XXS の serve・通常使用、RTX 4070 も同じ、RX 9060 XT は IQ3_XXS と Gyro-M）。
IQ3_XXS の generate では、B70 は upstream の `sycl/` 版より prefill が 1.8〜3.0 倍、decode が 36〜64% 速いです。ほかの型の 32K では prefill が 1.6〜1.9 倍、decode が 25% から 3.2 倍まで速いです（UD-Q4_K_XL の prefill は、ファイルがページキャッシュに載っていない起動直後の回で測れていません）。
RTX 4070 は IQ3_XXS の prefill が 16〜74%、decode が 0〜23% 速く、ほかの型の 32K では prefill が 8〜43%、decode が 1〜24% 速いです。RX 9060 XT は IQ3_XXS の prefill が 2.6〜2.8 倍、decode が 3〜20% 速いです。
serve の prompt は B70 で 1.8〜2.9 倍、RTX 4070 で 1.17〜1.73 倍、RX 9060 XT で 1.8〜2.6 倍速いです。serve の decode は B70 で 33〜64%、RX 9060 XT で −4〜+12%、RTX 4070 で −14〜+41% です（遅いのはどちらも温度 0.7 の 32K の 1 回だけです）。通常使用（chat-ja、chat-en、エージェント）では、3 枚とも最初のトークンまで、prompt、出力のどれも upstream より速いです。

移していないのは、DeepSeek V4.1-Flash への対応、アプリとターミナル、Docker、Mac、upstream の `sycl/`、
gfx12 の decode の取りこぼしの分け方の既定（`--pcie-frac 0.15`。RX 9060 XT で prompt が 4〜30% 遅く、XeStrata はリンクを測って分けます）、
推測の確からしさの門（`XESTRATA_SPEC_PROB`。前提の確率的な受理を 0.1.40 で取らなかったため）、
CUDA の機種別の表と融合カーネル（sm_80・sm_89 の MMVQ-IL の表、hyper-connection の広い読み、prompt の融合した IQ のエキスパート）、
GP100 の weight-stationary の MMQ と 4 ビットのエキスパートの形式（試験的で大きい）、hipBLASLt と `XESTRATA_SH_STREAM` の項目です。

## Strata 0.2.6 の取り込み

upstream の Strata 0.2.2 から 0.2.6（`4b9e05a2`）までの変更を Xe に移しています。
この間の upstream の変更の多くは、DeepSeek V4.1-Flash への対応（後の作業に回しました）、アプリ・ターミナル・Mac のもので、移していません。

移したものは次のとおりです。

- serve: `--conversation-cache-mib` を省くと、会話のキャッシュの大きさを RAM の空きから決める（Qwen3.8-Flash-Next と PlainModel の両方）。メッセージのないチャットの要求は 400 にする。1 トークンの返答には decode の速さを出さない。ツール呼び出しの別名と包みの境目を upstream に合わせ、閉じていない囲みのツール呼び出しも読む。ツールが返した画像では、止めてある画像のエンコーダーを起こす。画像のトークン数の上限に既定を持たせ、超えたものを断る。
- Web ページ: 書き換えた API キーを、ブラウザーの保存が断られてもすぐ使う。狭い画面で長い語・広い表・モデルの設定の行を画面の内に収める。
- setup: `--kv fp16` を受ける。入れたパッケージの記録に、書いた OS を残す。ビルドの道具は virtualenv のものを先に使い、古い cmake は無いものとする。2 分データが来ないダウンロードは、割合と次の一手を添えてそう言う。
- 推測デコード: `--spec-window auto` に、費用を見る窓の改良（文脈の長さごとの費用の表、外れた回を薄めて数える、1〜2 個先まで見て下書きを続けるか決める）を `XESTRATA_SPEC_COST` で選べるように。PlainModel の推測デコードも同じ部品を使います。1K では今の規則より狭い窓を選んで B70 で 2.7% 遅く、4K 以上では B70 と RTX 4070 のどちらでも今の規則と同じ窓を選びました（Qwen3.8 と Qwen3.6、2026-11-13。0.2.6 では 32K で 1.3〜3.2% 速かった）。既定（auto）は 6K 未満の文脈では切ります（`=1` で常に、`=0` で常に切る）。

この取り込みの間に XeStrata の側で変えたもの:

- Qwen3.8-Flash-Next と PlainModel（GLM・Ornith・llm-jp・Qwen3.6）で同じ働きの部分を、両方が呼ぶ部品にまとめました: オプションの解析、DONE 行、エキスパートの層の入れ替えの規則とプロファイルからの埋め方、`--adapt-every` を省いたときの間隔の自動の判定、PCIe の分担の測り方、`--prefill auto` の塊の選び方、prompt で CPU に回すエキスパートの候補、MTP の語彙の部分集合のドラフトヘッド。llm-jp-4.1 は入れ替えの間隔を自動で縮めません（縮めると遅くなるため）。PlainModel が VRAM の枠から選ぶ prompt の塊は、4096 より大きくするときだけ使います。
- PlainModel にも、`--vram-elastic`、`--low-ram on|mmap`、`--kv int8`（GLM を除く）、`--prefill-cpu-share`、`XESTRATA_RAM_ADAPT`、`--adapt-min-gain`、`XESTRATA_FETCH_THREADS`、`--spec-window` を通しました。GLM-5.3-Flash は DSA の潜在を RAM に置き、空いた VRAM をエキスパートに回します（`--kv-ram`、既定 on）。
- MTP の下書きのヘッドを、その GPU で速ければ 2 ビットの粗い積と上位 16 個の選び直しにします（Qwen3.8-Flash-Next と PlainModel の両方）。
- SSD から読むエキスパートを順位で飛ばす `XESTRATA_DISK_SKIP_FROM`（opt-in、Project Maya の #109 から）。
- 画像の層を流すときの写しを、スロットごとのリレーと複数スレッドの memcpy にしました。
- MTP の 2 ビットの下書きのヘッドで、文脈が 49152 トークン以上のときは、粗い値は候補の 16 個を選ぶだけにし、下書きの確率は選び直した 16 個から取ります（残りの約 10 万語の粗い値の誤差が確率を下げ、長い文脈で下書きが早く止まっていました: RTX 4070 の 64K の serve の最初のターンが 56.5 → 71.7 tok/s）。短い文脈では今までどおりです（残りを確率 0 にすると確率が高く出すぎ、RX 9060 XT の 32K で 5% 遅くなりました）。
- serve: 会話のキャッシュを RAM の空きから決める既定では、チャットの次のターンのたびに今の会話を RAM に退避していました（クライアントが前の返事の考えを履歴から外して送るので、切り替えとして扱われていた）。同じ会話の次のターンでは退避せず、別の会話に切り替えるときだけ退避します（Qwen3.8-Flash-Next と PlainModel の両方。B70 で 1 ターンあたり 80〜140 ms、RX 9060 XT で 120〜370 ms の待ちがなくなりました）。
- 起動時に経路を測って選ぶ 3 か所を、ぶれや VRAM の不足で遅い方を選ばないようにしました。DeltaNet の 2 つの形は塊の先頭 1024 トークンまでで測ります（16384 トークンの塊では測りの写しが VRAM に入らず、5 倍遅い形のままでした: RTX 4070 の serve の 32K の prompt が 1,687 → 1,958 tok/s）。エキスパートの積を FP16 にするのは int8 より 3% 以上速いときだけにします。hyper-connection の読み方は 3 つの形を 3 回交互に測ります。

upstream v0.2.6 と 3 社の GPU で比べました（B70 は IQ3_XXS・Q2_0・IQ2_XS・IQ3_S・UD-Q4_K_XL の generate、RTX 4070 は IQ3_XXS の serve、RX 9060 XT は IQ3_XXS の generate と serve）。
IQ3_XXS の generate では、B70 は upstream の `sycl/` 版より prefill が 1.8〜2.7 倍、decode が 24〜48% 速いです。ほかの型の 32K では prefill が 1.7〜2.5 倍、decode が 12% から 2.9 倍まで速いです。
RX 9060 XT は prefill が 2.1〜2.8 倍、decode が 8〜10% 速いです。
serve の prompt は、RTX 4070 で 1.16〜1.75 倍、RX 9060 XT で 1.9〜2.6 倍速いです。serve の decode は、RTX 4070 で 0.2〜52%、RX 9060 XT で 1〜24% 速いです。通常使用（chat-ja、chat-en、エージェント）では、3 枚とも最初のトークンまで、prompt、出力のどれも upstream より速いです。

移していないのは、DeepSeek V4.1-Flash への対応、アプリとターミナル、Mac、upstream の `sycl/`、DFlash2 のドラフター、
CPU の i-quant の AVX-512 の行（本体に効くのは IQ3 の 1 トークンだけで、DeepSeek の CPU の経路の一式に乗るため。AVX-512 の機械がなく速さを測れません）、
trellis の新しい型（TQK5X・TQK4X ほか、実験的で配布のモデルがありません）、gfx1151 の HIP の実験、
推測デコードの起動直後の入れ替え（upstream 自身が opt-in に戻しました）、非同期の段を測って選ぶ形（XeStrata の段では、非同期にしても RTX 4070 の serve で速くなりませんでした）、
hipBLASLt の調整表（XeStrata の AMD の積は oneMath の rocBLAS です）です。

## Strata 0.2.2 の取り込み

upstream の Strata 0.2.1.1 から 0.2.2（`011c9e60`）までの変更を Xe に移しています。
この間の upstream の変更の多くは、アプリ・ターミナル・Mac・upstream の `sycl/` のもので、移していません（0.2.0-beta と同じ理由）。

移したものは次のとおりです。

- CPU のプール: ワーカーの置き場所を起動時に言い、`--pool-cores`（実験的）で置き場所を指定できる。エンジンのスレッドに名前を付ける。serve は設定の `pool_cores` をエンジンに渡す。
- 推測デコード: 開始の規則を `spec_loop_runs` にまとめ、`XESTRATA_SPEC_WATCH` で止まりを見張る。
- serve: prefill の短い残りの窓で logits が非有限になったら、次の要求は prompt 全体をまとめて読む。エンジンを起動できないときは、接続を切らずに理由つきの 503 を返す。`"api_key": "none"` と `--no-api-key` でキーなしを選べる。
- serve: Anthropic の usage で、次の要求のために残したトークンを `cache_creation_input_tokens` として返す。llama.cpp の `thinking_budget_tokens` も思考の上限として読む。1 トークンずれた囲みのツールの呼び出しを読み、宣言していないツールの呼び出しは文として返す。done の行の速さはエンジンの decode の時間から取る。
- setup: SHA-256 が合わないダウンロードは消さずに違う部分だけ取り直し、大きなファイルの確かめにも進み具合を出す。`--gguf-dir` の利用者のファイルは消さない。空き容量は入り切らないときだけ止める。ごみ箱の中のフォルダーを前のインストールとして覚えない。
- iq_pack: experts.bin の書き込みが途中で止まったら、書き終えた層を残して次の層から続ける。

この取り込みの間に XeStrata の側で直したもの:

- intel/llvm v7.2.0 での後退: XMX の積と grouped の積の `joint_matrix_prefetch` を外す（IGC 2.28 が v7.2.0 の先読みを扱えず、B70 の 1K の prefill が 468 → 751 tok/s）。AMD 向けの libspirv に GroupNonUniform の shuffle を足す（パッチ `hip-group-shuffle`）。iq_mmq の作業グループの大きさを属性で宣言し、AMD でのレジスターの退避をなくす（RX 9060 XT の 32K の prefill が 1,113 → 1,250 tok/s）。
- 推測デコード: 窓のグラフを作った回の時間を窓の選び方の測りに入れない。測っていない大きさの窓の時間は、いちばん近い測った小さい大きさから見積もる。Gyro の trellis の層では、VRAM の入れ替えの位置で窓を切らない（Gyro-M の decode が B70 の 1K で 48.0 → 54.2 tok/s）。グラフのない大きさの窓は、VRAM が足りなければ収まる大きさに縮める（B70 の 262K で落ちなくなった）。

upstream v0.2.2 と 3 社の GPU で比べました（generate は IQ3_XXS と Gyro-M を 3 枚、Q2_0・IQ2_XS・IQ3_S・UD-Q4_K_XL を B70 と RTX 4070、IQ3_XXS の serve と通常使用）。
IQ3_XXS の generate では、B70 は upstream の `sycl/` 版より prefill が 1.8〜3.0 倍、decode が 12〜61% 速いです。
RTX 4070 は prefill が 1.02〜1.16 倍、decode が 25〜37% 速いです。RX 9060 XT は prefill が 2.1〜2.7 倍、decode が 2〜7% 速いです。
Gyro-M の generate では、B70 は prefill が 1.3〜2.0 倍、decode が 8〜11% 速いです。RX 9060 XT は prefill が 1.3〜2.3 倍、decode が 1〜2% 速いです。
RTX 4070 は prefill が 1.08〜1.12 倍で、decode は 32K で同等、1K で upstream の文をなぞらせると 3% 遅く、XeStrata 自身の文では 17% 速いです。

移していないのは、CPU のプールを NUMA のノードごとに分ける形（XeStrata のプールにはノードの土台がありません）、アプリとターミナル、Mac、upstream の `sycl/`、
CUDA の Gyro の decode で trellis のコードブックを共有メモリに置く形と、その Pascal 向けのカーネル（XeStrata の SYCL のカーネルで試し、RX 9060 XT で 17〜19% 遅く、B70 と RTX 4070 でも速くなりませんでした）、HIP の gfx11 と gfx1151 だけの直しです。

## Strata 0.2.1.1 の取り込み

upstream の Strata 0.2.0-beta から 0.2.1.1（`9ae93f7b`）までの変更を Xe に移しています。
この間の upstream の変更の多くは、新しいアプリ・ターミナル・Docker・Mac のもので、移していません（0.2.0-beta と同じ理由）。

移したものは次のとおりです。

- setup: Meteor Lake・Arrow Lake-H の LP-E コアを E コアに数える。低 RAM の収まりの判定で、下書きの層の VRAM（約 1 GB）を数える。
- ダウンロード: 既定 16 本の接続、1 つの要求で隣の 4 片、長さを返さない HEAD には 0-0 を聞く。MTP の取得は切れやすい回線でも続き、setup が止まれば止まる。
- setup: 新しい `--data-dir` はモデルのファイルを勝手に動かさず、前の置き場所も見つけ続ける（`--move-data`、`--set-default-data-dir`）。
- serve: エンジンが signal 9 で終わった理由に、有限の RLIMIT_RTTIME（rtkit のない PipeWire）を挙げる。
- Windows: `xestrata-engine.exe`・`xestrata-image.exe`・`xestrata-device.exe` に UTF-8 のマニフェストを付け、ANSI でない文字のパスを通す。
- `--batch`: 2 枚の NVIDIA の GPU の層の分割では、既定のグループを 1 つの直列にする（upstream の計測で +12〜18%）。

RX 9060 XT で upstream v0.2.1.1 と比べました（IQ3_XXS の generate）。prefill が 2.1〜2.7 倍、decode が 3〜7% 速いです。
upstream のエンジンの変更は、ほかの GPU の計測の経路（1 枚の GPU、エキスパートを RAM に置く）に効かないので、B70 と RTX 4070 は 0.2.0-beta の比べを使っています。

移していないのは、アプリとターミナル、Docker の配布、Mac、upstream の `sycl/`、HIP の内蔵 GPU と hipBLASLt の表と GPU のキューの開き方（RX 9060 XT の XeStrata の prompt は起動ごとに 1.5% の幅に収まっていて、upstream の直した 2 つの速さに分かれる現象は出ていません）、
experts.bin の pack の I/O の先読みとファイルの段（その土台が XeStrata にありません）、Intel の GPU の画像の読み手（XeStrata は自前の視覚エンコーダーで読みます）です。

## Strata 0.2.0-beta の取り込み

upstream の Strata 0.1.45 から 0.2.0-beta（`3d1fb480`）までの変更を Xe に移しています。
正式な 0.2.0 が出たら、その差分を足します。

移したものの主なものは次のとおりです。

- Gyro-S と Gyro-M（agentionai）: trellis 量子化の型（TQ2_T・TQK6・TQK7）と、エキスパートの Hadamard の回転。CPU（AVX2・AVX-512）と GPU（decode と prompt）で計算し、setup の一覧に載せる。pack の版は xs2。
- setup: ダウンロードするモデルのファイルを大きさと SHA-256 で固定する。大きなファイルは複数の接続でダウンロードする。ビルドの道具（cmake・ninja）は実行時の requirements から分ける。SSD でも RAM に余裕があれば `--ple-io ram` を付ける。画像を読めるモデルでは画像を既定で入にする。
- serve: 文脈に収まらない `max_tokens` は、400 にせず残りの余地まで縮める（`--no-fit-max-tokens` で以前のように断る）。
- CPU のプールの既定のワーカーを、32 本とコンテナの CPU の割り当てまでにする。cgroup の RAM の判定では、きれいなページキャッシュを回収できるものに数える。
- 適応の段で入れ替えたエキスパートは、既定で 2 つ目の窓から使う（`XESTRATA_ADAPT_LAG`、RTX 4070 で decode +8.6%）。
  `auto`（既定）は、入れ替えのコピーを待つ窓がある最初の数窓で「次の窓が待つ」と「既定の窓数」を交互に回し、トークン当たりの時間が 3% 以上短い方を選ぶ（`0` は次の窓が待つ、`2` 以上はその窓数）。
- 画像の読み手（xestrata-vision）は、文章のモデルの語彙とヘッダーだけを読む（Gyro の型のファイルでも開ける）。

この取り込みの間に XeStrata の側で直したもの:

- VRAM のエキスパートの入れ替えの位置で窓を切るのは、CPU と GPU で丸めが違う層があるモデルだけにする（`XESTRATA_ADAPT_CUT`。`auto` が既定で形式の規則、`0` と `1` で固定）。Qwen3.8 の 4 つのファイルはすべての層が CPU と GPU で同じビットになるので、窓を切らない。同じ文で decode は RTX 4070 で 59.3 → 61.3、B70 で 80.5 → 86.9 tok/s。

upstream v0.2.0-beta と 3 社の GPU で比べました（generate は 5 つのモデルファイル、serve、通常使用、画像の生成）。
IQ3_XXS の generate では、B70 は upstream の `sycl/` 版より prefill が 1.7〜3.0 倍、decode が 18〜38% 速いです。
RTX 4070 は prefill が 1.02〜1.14 倍、decode が 23〜34% 速いです。RX 9060 XT は prefill が 2.1〜2.7 倍、decode は同等から 8% 速いです。
画像の生成（1024x1024、8 ステップ）は、XeStrata が B70 で 7.6 秒、RTX 4070 で 13.9 秒、RX 9060 XT で 22.5 秒、upstream がそれぞれ 52.0 秒、26.6 秒、73.7 秒です。

移していないのは、upstream の新しいアプリ（`apps/`）とターミナル（`strata_cli`・`strata_control`）、配布の変更（インストーラー、署名つきの配信、エンジンの部品の更新）、NVFP4、upstream の `sycl/`、
CUDA と HIP だけの速さの経路（sm_120 の切り替えなど）、HIP の仮想メモリでの `--kv-grow`（intel/llvm の HIP のアダプターに仮想メモリの拡張がない）、
GGUF をその場で読む省 RAM の形（写さずに使うエキスパートと I/O の先読み）、upstream の試験の道具と記録です。

## Strata 0.1.45 の取り込み

upstream の Strata 0.1.44.1 から 0.1.45（`477ac13c`）までの変更を Xe に移しています。

移したものの主なものは次のとおりです。

- prompt の MoE の積が VRAM に常駐するエキスパートをキャッシュの枠から直接読む（B70 の 1K で prefill +13%）。入力の量子化はトークンごとに 1 回にし、k 行に配る。
- 確認の窓の QSA の採点はブロックの鍵を 1 度読んで窓の行を全部まかなう（RTX 4070 の 32K で decode +1.3%）。
- エキスパートの decode で、活性を作業グループのローカルメモリに置く形を、入るすべての GPU で既定にする（B70 で +1%）。
- QSA の top-k の基数の桁を協調の走査で選ぶ。常駐のエキスパートの層は MMQ のグループを行の数の多い順に組む。
- 層の分割の prefill で、チャンクの行を 8 つに分けて次の GPU に渡し、2 枚の転送を重ねる（`STRATA_HANDOFF_PIPE=0` で 1 回の転送）。
- 会話のキャッシュ: 新しい会話が別の会話の途中のチェックポイントで始まるときは借りる（#1164）、少ししか伸びていない会話は置き直さない（`--conversation-cache-park-gain`）、チェックポイントを会話の終わりから間を空けて残す（`--prompt-cache-spread`）。
- RAM 常駐の形: 予算に余りがあれば、prompt の経路が借りる枠のエキスパートも RAM に置く（#1683）。
- `--layer-split auto`: 2 枚で小さな段ができたら置き場所の広さの割合で選び直す（#2644）。
- serve: 思考の上限（`max_tokens` と Anthropic の `budget_tokens`）、sampling のない要求にモデルカードの値、プロンプトを最後に共有した前置きからトークン化する（#1852）、次のターンを会話のチェックポイントを持つスロットへ送る（#2518）、短い prompt を先に受け入れる（`XESTRATA_ADMIT_SHORT_FIRST`、opt-in）、パーサーの修正。
- opt-in のまま移したもの: `XESTRATA_RAM_ADAPT`。
- `XESTRATA_EXCHANGE_ROTATE` は、層ごとにエキスパートの大きさが違う pack で、予備のぶんの RAM（IQ3_XXS・`--adapt-swaps 80` で 1.20 GiB）の余裕があるときに自動で入ります（`0`・`1` で固定）。

この取り込みの間に XeStrata の側で直したものは次のとおりです。

- int8 と FP16 のエキスパートの積の境目を、1 つのエキスパートに 1024 行まで測る。256 行までしか測らず、IQ3_S と UD-Q4_K_XL の 32K のチャンクを int8 で読んでいた（B70 の IQ3_S の 32K の prefill 1,896 → 2,425 tok/s）。
- 大きな prompt のチャンクを取るかの判定で、もう置いた RAM の写しを数え直さない（B70 の UD-Q4_K_XL の 32K の prefill 1,760 → 2,412 tok/s）。
- 層の分割の後ろの段はキャッシュから一律に 1 GiB を引いていたのをやめ、確認の窓の分と、最後の段で後から載る下書きの head の分だけを残す（10 GB に絞った B70 + RTX 4070 で 2 枚目のキャッシュ +560〜700 枠、prefill +4〜5%）。

upstream v0.1.45 と 3 社の GPU で比べました（generate は 5 つのモデルファイル、serve、通常使用、画像の生成）。B70 は今回から、upstream の `sycl/` 版とも比べています。
RTX 4070 は generate の prefill が 5 つのファイルのすべてで同等以上（IQ3_XXS の 1K と IQ2_XS は 17%、UD-Q4_K_XL は 1% 速い）、decode が 13〜30% 速いです。
RX 9060 XT は prefill が 2.1〜3.2 倍で、0.1.44.1 では 2.5〜4.7% 遅かった IQ3_XXS の decode も同等から 7% 速くなりました。
B70 は upstream の `sycl/` 版より prefill が 1.6〜2.5 倍、decode が 15〜95% 速いです（UD-Q4_K_XL は `sycl/` 版の答えが壊れていて比べていません）。
画像の生成（1024x1024、8 ステップ）は、XeStrata が B70 で 7.6 秒、RTX 4070 で 13.2 秒、RX 9060 XT で 22.3 秒、upstream の sd-cli がそれぞれ 52 秒、25.5 秒、73.5 秒です。
取り込みの前の XeStrata と同じ文で比べると、decode は B70 と RTX 4070 で同じ速さ、prefill は B70 の 1K が 35% 速くなりました。

移していないのは、upstream の `sycl/`、バッチの改善（8 を超えるスロットの層の分割、スロットの GPU 間のコピー、受け入れの opt-in、行をまとめた起動）、Turing（sm_75）の速さの経路、
`--prefill-pipe`（B70 + RTX 4070 では小さなチャンクで prefill が 38.6% 遅い）、`XESTRATA_SPLIT_OVERLAP`（XeStrata の段の受け渡しは GPU どうしで同じメモリの語を見られない）、
計測のない CUDA の opt-in の経路、upstream の試験の道具と記録です。

## Strata 0.1.44.1 の取り込み

upstream の Strata 0.1.43.1 から 0.1.44.1（`296b59d2`）までの変更を Xe に移しています。
今回から、Windows の変更も取り込む対象にしました（XeStrata の Windows 版にない HIP の部分を除く）。

移したものの主なものは次のとおりです。

- Qwen3.6-35B-A3B（Unsloth の MTP の層付きの UD-Q4_K_XL・UD-Q6_K_XL・Q8_0）を、Ornith-1.5 と同じ plain の経路で読む。
  upstream は Qwen3.6 に CUDA 専用のエンジンを作りましたが、XeStrata では F32 の GDN の alpha・beta と MTP の層の BF16 の router を読めるようにしました。
  llama.cpp と答えを比べ、3 社の GPU で動かしています。prompt の直後に、その prompt が多く使ったエキスパートで VRAM の層を一度に埋める（upstream の q36 の考え方）ので、decode が 20〜40% 速くなりました（Ornith と llm-jp にも効きます）。
- 画像の生成（Qwen-Image-2.1-Turbo、`POST /v1/images/generations`）。upstream の stable-diffusion.cpp の代わりに、XeStrata 自身の SYCL の画像のエンジン `xestrata-image` で作ります（DiT・VAE・文の読み手、行列ユニットの注意。サーバーの中で常駐し、VRAM に入らない層は流し込む）。
- Unsloth のファイル: K-quant のエキスパートの重みを、グループの呼び出しで 1 回だけ展開する（KqSplit、同じビットの答え。B70 の UD-Q4_K_XL の decode +7%）。setup は UD-Q5_K_XL・UD-Q6_K_XL・Q8_0 も選べます（experimental）。
- 確定のグラフで GDN と QSA の層をまとめて 1 回の起動にする（`XESTRATA_COMMIT_BATCH`、既定でオン）。
- 層の分割の自動の配置（近い同点では窮屈なカードに余裕を残す、遅いリンクの重みは 4 倍の差から）、起動時の VRAM の空きの確認を後ろのカードにも。
- prompt の塊を 16384 にするのは、RAM が貸し出しを賄えるときだけ。arena の MADV_HUGEPAGE は、空いている 2 MiB のブロックが半分に満たないときは頼まない（#2126）。
- serve: 長い読みが短い要求に次の塊で譲る（#656）、opencode の `<env>` を最初のユーザーの発言へ移す（`"env_to_user"`、opt-in）、画像があるときだけ画像のエンコーダーを動かす（`"elastic"`、opt-in）、
  GPU のランタイムの立ち上げで止まったエンジンの起動し直し、落ちたエンジンの終わり方の表示、Responses API の `reasoning_effort`、会話のキャッシュの画像の照合（#2228）、stdin の読み手の終了の競合。
- 量子化の品質の物差し（`tools/quant_quality.py`、`tools/quant_formats.py`）。
- `XESTRATA_HC_Q8`（ハイパーコネクションの down/up を Q8_0 で読む、BF16 の pack では読み込み時に Q8_0 にする）: 読み込み時に両方の形を測って速い方を使い、Q8_0 なら GPU 1 枚のときは BF16 の写しを VRAM から外します。AMD では次の射影のための q8_1 も読みの中で書きます（B70 で decode +3.5%。答えが変わる近似）。
- opt-in のまま移したもの: `XESTRATA_COPY_PRIO`（自動は 0。1 で `--pcie-mode dma` のコピーの stream を高優先にする）、`XESTRATA_DENSE_F32`。
- 起動時の短い測りで選ぶもの（`0`・`1` で固定できる）: CPU プールの待ち方（`XESTRATA_POOL_UMWAIT`。CPU の WAITPKG を確かめ、公開から起きるまでの時間をスピンと UMWAIT で測ります）と、PCIe の分担の下限（`XESTRATA_PCIE_MIN1`。分担を固定したときに 1〜2 個の外れが 0 に丸まらないよう、CPU・PCIe・GPU の 1 個あたりの時間を測って決めます）。

この取り込みの間に XeStrata の側で直したものは次のとおりです。

- GPU の問い合わせの失敗を未完了と区別し、待ちをエラーで抜ける（issue #12）。RAM is tight の警告を、エキスパートを実際に RAM に置く量で出す（issue #11）。
- WSL2: setup が GPU を見つけない、plain の経路の PCIe の分担での失敗、prompt のバッファの不足（報告の 3 件）。
- Podman（podman compose）で入れて動かすコンテナ。

upstream v0.1.44.1 と 3 社の GPU で比べました（generate は 5 つのモデルファイル、serve、通常使用）。
RTX 4070 は generate の decode が 5 つのファイルのすべてで 8〜21% 速く、prefill は同等以上（IQ3_XXS の 1K は 16%、IQ2_XS は 15% 速い）です。
RX 9060 XT は prefill が 2.1〜3.2 倍ですが、IQ3_XXS の decode は 2.5〜4.7% 遅くなりました。
upstream が gfx12 で hyper-connection の down/up を Q8_0 で読むのを既定にした分（upstream の側で切って比べると +4.4%）です。XeStrata も読み込み時に測って Q8_0 を選びますが、RX 9060 XT での伸びは +0.6% にとどまります。
B70 は upstream と比べず、取り込みの前の XeStrata と同じ文で比べて同じ速さです。

移していないのは、Mac（Metal）、upstream の `sycl/`、NVFP4 のエキスパート（FP4 の行列ユニットのない GPU では IQ 系より prompt が遅く、品質も UD-Q4_K_XL より下）、
upstream の Qwen3.6 の CUDA 専用のエンジン、gfx11 の WMMA への gfx1103 の追加（RDNA2・3 の保留と一緒に）、`--prefill-head-alias` と arena の pin by need（XeStrata の USM と arena の作りに合わない）、
upstream の試験の道具と記録です。

## Strata 0.1.43.1 の取り込み

upstream の Strata 0.1.43 から 0.1.43.1（`a0116b27`）までの変更を、Windows の実装を除いて Xe に移しています。
取り込みは 1 件ずつ手で移しました。

移したものの主なものは次のとおりです。

- ページ固定していない RAM から非同期に GPU へ送るコピーを、ピン留めした枠か送り終わるまで待つ形にする（upstream #1996）。
  当てはまったのは、下書きの K/V のトークン、MTP の prompt の記録、画像の埋め込みの行です。
- PLE: 直接読みが遅いと測れても mmap には切り替えず警告だけにする（`STRATA_PLE_AUTO_MMAP=1` で 0.1.42 の切り替え）。
  `--ple-io mmap` の prompt の読みは次の 4096 行を先に読ませる（upstream #2040・#1984・#1971）。
- 見張り: 10 秒以内に動いたファイルの層は I/O の猶予を保つ。会話の退避は 1 MiB ずつ写して見張りに止められないようにし、
  停止の報告に MTP の下書きの待ちを名前で出す（upstream #2049・#2065）。`XESTRATA_TRACE=2` は起動の印ごとにデバイスを待つ。
- serve: BGEN の枠を断られたら別の枠を試す（#1994）、before_load のフックの失敗に 503 と Retry-After を返し `/unload?wait=1` を受ける（#1997）、
  文中の `</tool_call>` を文として扱う（#2026）、`<tool_call>` の中の Markdown の囲みの呼び出しを読む（#2037）、
  画像の準備中に去ったクライアントの準備を止める（#1999）。
- 層の分割と `--batch` では、既定の組を `--batch-mtp` に譲らない（#2038、`XESTRATA_BATCH_MTP_SERIAL` の `auto`（既定）は流れ作業、`1` で前の形）。
- setup の `--calibrate` は `--model`・`--family` のモデルを調整する（#2036）。ベンチマークはエンジンが使う GPU だけを報告し（#1985・#2062）、
  ベンチマークが起動するサーバーは `--expert-profile-save` なしで動く（#2088）。

この取り込みの間に XeStrata の側で直したものは次のとおりです。

- prefill: FP16 と int8 のエキスパートの行と組を同じ場所に置き、両方の経路を持つモデルで塊が縮まないようにする。
  RTX 4070 の UD-Q4_K_XL の 32K で 460 → 617 tok/s（upstream は 620）。
- 小さい VRAM の prefill: hyper-connection の行を注意と MoE の作業領域の中に置き（`XESTRATA_PREFILL_HC_IN_REGION=0` で元の置き方）、
  FP16 のエキスパートの入力の行を down の出力の行の上半分に置く（`XESTRATA_PREFILL_XS_IN_DM=0`）。
  RTX 4070 を 8 GB に絞った IQ3_XXS の 64K で 291 → 428 tok/s。
- `--spec` でエキスパートキャッシュが 0 枠のときは、prompt を読む前に止める。
  setup は、KV キャッシュとエキスパートキャッシュが VRAM にも RAM にも入らない文脈の長さを警告し、入る `--context` か `--kv q4_0` を示す。

upstream v0.1.43.1 と 3 社の GPU で比べました（generate は 5 つのモデルファイル、serve、通常使用）。
RTX 4070 は IQ3_XXS の decode が 1K〜262K で 11〜20% 速く、prefill は同等以上です。
RX 9060 XT は prefill が 2.1〜3.1 倍、decode が 5〜12% 速くなりました。
8 GB のカード（VRAM の残りを 8 GB にして両方を動かす）でも、prefill は RTX 4070 の 64K で 270 対 194 tok/s、RX 9060 XT の 32K で 713 対 306 tok/s です。

移していないのは、Windows の実装、upstream の `sycl/`、`XESTRATA_ADAPT_OVERLAP` とその callback のスレッドの調整（XeStrata にない機能）、
適応の段のコピーの着地を host の関数でなく stream の順番どおりのコピーで知らせる変更（XeStrata で host の関数でフラグを立てるのは `--pcie-mode dma` のときだけで、既定の経路は使わない）、
塊を小さくすると貸す枠が増える場合の扱い（#2090。XeStrata のリングは塊について単調）、upstream の試験の道具と記録（`bench/results`）です。

## Strata 0.1.43 の取り込み

upstream の Strata 0.1.42 から 0.1.43（`7695859a`）までの変更を、Windows の実装を除いて Xe に移しています。
取り込みは 1 件ずつ手で移し、速さの経路は考え方ごとに 1 つの GPU で A/B を測って、速いか差がなければ入れました。

移したものの主なものは次のとおりです。

- logits が NaN や無限大になった窓を見張り、その要求を終えて使い回す状態を捨てる仕組み（upstream #879、既定。`XESTRATA_NAN_GUARD=0` で切る）。
- prefill: K/V のホストの写しを 1 レーン 4 バイトで書く（`XESTRATA_KV_WIDE_HOST=0` で以前の形。B70 の 64K で 1,974 → 2,399 tok/s）、
  エキスパートのコピーを GPU のコピー専用のエンジンで流す（Level Zero、`XESTRATA_COPY_ENGINE=0`）、
  1 つのエキスパートの gate/up と down を 1 回の起動で逆量子化する（`XESTRATA_PF_DQ_FUSED=0`。B70 の UD-Q4_K_XL の 32K で +9%）、
  塊の step 記録とトークン id を page-lock したバッファから送る（upstream #1712）。
- decode: エキスパートの足し込みを複数行の combine に畳む（`XESTRATA_FUSE_HITS=0`）、Q3_K の GEMV の展開をブロックごとに 1 回にする、
  route-resident の入れ替えを 1 トークン 1 sub-group にする（opt-in の機能の中、B70 で +3.9%）。
- serve の tail skip を、エキスパートの 90% 以上が VRAM にあるときは既定で切る（upstream #1884、`XESTRATA_TAIL_SKIP_MAX_CACHED`）。
- serve の報告（単独の経路とバッチの枠を移った要求の時間とトークン数、`timings.prompt_n`。upstream #1869・#1870）、
  `XESTRATA_DROP_CACHE_ON_EXIT`、RAM が足りないときのファイルキャッシュの案内、`XESTRATA_TRACE` の VRAM の残り。
- upstream のベンチマークのモード（`strata_bench.py`）を、そのままの測り方で動かす `xestrata --setup --bench-original-compat`（[BENCH](BENCH.ja.md)）。

この取り込みの間に XeStrata の側で直したものは次のとおりです。

- 外れたエキスパートの CPU と PCIe の分け方で、CPU の時間を decode の実測で詰め直す（[外れたエキスパートの CPU と PCIe への分け方](#外れたエキスパートの-cpu-と-pcie-への分け方)）。
- RAM がエキスパートの全体に足りないときは、全体を読み込まずに `--resident-experts` と同じ形で起動し、ホスト USM へ移すのは層ごとに RAM の空きがあるときだけにする（issue #10、[DETAILS](DETAILS.ja.md)）。
- エンジンが例外で止まったときに、何をしていたか、SYCL と Unified Runtime の番号、VRAM の空き、GPU が応答するかを 1 行で報告し、Web アプリにも出す（issue #10）。
- Web アプリの GPU の情報を 3 社でそろえる（Intel の温度は GPU 本体、層の分割の 2 枚目以降、製品名、PCIe の経路の上限）。
- serve で、発言の始まりのチェックポイントの RAM への書き込みを返答の後に回す（[DETAILS](DETAILS.ja.md)）。RTX 4070 で続きの要求の待ちが 1K 288 → 222 ms、64K 317 → 253 ms。

upstream v0.1.43 と 3 社の GPU で比べました（generate は 5 つのモデルファイル、serve、通常使用）。すべての項目で upstream と同等以上です。
RX 9060 XT は decode が IQ3_XXS の 1K〜262K で 9〜13% 速く（xe0.1.42 からは 25〜45% 速く）、prefill は 2.9〜3.5 倍です。
RTX 4070 の serve の続きの待ちは 222 ms と 253 ms で、upstream の 240 ms と 282 ms より短くなりました。

移していないのは、Windows の実装、upstream の `sycl/`、HIP と CUDA の専用のカーネルの調整表と試せる機器のない GPU のための経路（Pascal の行の表、gfx12 の prompt の融合 int8 のエキスパート、gfx12 の decode の HC norm とフラグ待ちの畳み込み、HIP の層分割の既定）、
upstream が `sycl/` 向けに作った混ぜたエキスパートのプロファイル（generate の 32K の decode が B70 で −2.0%、RTX 4070 で +6.2%、RX 9060 XT で −5.3%）、
XeStrata で効く所のないもの（`XESTRATA_ADAPT_OVERLAP`、PcieAuto の割合、GEMM の operand の段取りを省く変更、貸した枠を DMA で埋め直す変更）、upstream の記録（`bench/results`）です。

## Strata 0.1.42 の取り込み

upstream の Strata 0.1.40.2 から 0.1.42（`61b3fb5d`）までの変更（0.1.41 を含む）を、Windows の実装を除いて Xe に移しています。
取り込みは 1 件ずつ手で移しました。

移したものの主なものは次のとおりです。

- serve の decode で、外れのエキスパートを順位 7 以下で選んだトークンではその寄与を 0 にし（窓ではなくトークンごとに決める）、検証の窓のどのトークンもその順位でしか選ばないものは読まず計算もしない `STRATA_ROUTE_TAIL_SKIP=7`（opt-in）。XeStrata は 3 社の GPU で KL・タスクの確認・速さを測り直して全機種で既定にしていましたが、upstream が答えの質を理由に既定をオフにしたので（962475995）、既定をオフに戻しました。
- 検証の窓で VRAM にあるエキスパートに寄せる `STRATA_ROUTE_RESIDENT`（opt-in）と `STRATA_ROUTE_PRIOR`（opt-in）。どちらも答えが変わる。
- prefill の CPU 分担の拡張（ページ固定でない RAM のエキスパートも渡す）と、全エキスパートが VRAM にあるときやメモリが足りないときに既定で使わない規則。
- `--aux-cpus`（opt-in）、PLE の `--ple-io direct` が遅いときの mmap への切り替え、stager の RAM 向けの読み方、埋め込みのバッファの使い回し。
- メモリの見張り（cgroup の上限と `--memory-limit-mib`）、会話キャッシュの `--conversation-cache-min-tokens`、`--session-save-reclaim`、`logit_bias`、`xestrata-engine --version`。
- serve の変更（Prometheus の形のメトリクス、Monitor の会話キャッシュのカード、AMD の GPU の読み取り、サンプリングの欄の扱い、道具の一覧の順など）と、setup の変更（設定の書き方、`--gguf-dir`、回転ディスク、画面を出すカードの VRAM の予約、ハイブリッド CPU のワーカー数など）。

upstream v0.1.42 と 3 社の GPU で比べました（generate は 5 つのモデルファイル、serve、通常使用）。
B70 と RTX 4070 はすべての項目で upstream と同等以上です。RX 9060 XT は generate の 32K の decode（36.9 対 44.6 tok/s）と serve の 1K の続きの待ち（526 対 478 ms）を除いて同等以上で、prefill はどの長さでも 2.2〜2.4 倍です。
RX 9060 XT の decode は同じ条件でも回ごとに 3 割ほど変わり（upstream も同じ）、原因はまだ分かっていません。

最終の比較の途中で見つけて、XeStrata の側で直したものは次のとおりです。

- RAM に置いたエキスパートの写しのページ固定が 1 回の確保で断られたら（B70 の Level Zero は 30 GiB まで）、エキスパートの境目で分けて確保する（`XESTRATA_PIN_PARTS=0` で前の形）。B70 の UD-Q4_K_XL の 32K で prefill +17%、decode +14%。
- 外れたエキスパートの CPU と PCIe への分け方の測定をそろえる（[外れたエキスパートの CPU と PCIe への分け方](#外れたエキスパートの-cpu-と-pcie-への分け方)）。
- serve の短い読み込みで、窓とまとめ読みの実測を読み始めの位置と一緒に覚え、近い位置の実測だけで選ぶ。試しのまとめ読みは 16K までの位置でだけ行う。RTX 4070 の 64K で、4 ターン目の続きが 505 ms から 304 ms に。

速くするための経路は、B70・RTX 4070・RX 9060 XT のどれかで速くなり、ほかで遅くならないものを移しました。
CPU 分担の auto の判定（upstream f2a3b946）は B70 と RTX 4070 の 1K で遅く（−1.3%・−4.4%）、plain の Ornith 32K でも差が無かった（−0.1%）ので消しました。Q8_0 の逆量子化の 4 分け（upstream e910e410）はどちらでも速くならなかったので外しました。
移していないのは、Windows の実装、upstream の `sycl/`（同じ B70 で速さを比べる相手）、HIP と CUDA の専用のカーネルの調整表、試せる機器のない GPU のための経路の一部（RDNA2 特有のもの）、複数 GPU の変更（別の作業でまとめて移します）です。

## Strata 0.1.40 の取り込み

upstream の Strata 0.1.39 から 0.1.40.2（`e8ca9afd`）までの変更を、Windows の実装を除いて Xe に移しています。
AMD（HIP）の変更は XeStrata の AMD の経路（同じ SYCL のカーネル）に当てはまるものを移しました: AMD の APU の統合メモリ（setup と、エンジンの内蔵 GPU の空き）と、全機種向けだった融合した RMSNorm + RoPE です。
upstream の HIP 専用の速くするためのスイッチは、RX 9060 XT（gfx1200）で入り切りして測り、gfx11（RDNA3）と gfx103x（RDNA2）専用のものは試せる機器がないので移していません。
upstream の `sycl/`（Intel の GPU 向けの別の移植）はコードに入れず、同じ B70 で速さを比べる相手にしています。
NVIDIA の GPU では contrib-llvm のビルドで同じカーネルが動きます（[BUILD](BUILD.ja.md)）。

移したものの主なものは次のとおりです。

- 会話をファイルに保存して戻す（セッションファイル）、使い捨ての要求でチェックポイントを取らない `ckpt=0`、共有の先頭を固定する `pin=N` と `tools/research_run.py`。
- PLE の表は GGUF にある形式のまま読みます: IQ4_NL（既定の表）、Q4_0、Q5_0、Q5_1、Q8_0、FP8（E4M3 とスケール）、BF16。
  upstream の測定では、BF16 の表と比べた行ごとの平均誤差は Q8_0 0.53%、FP8 2.64%、Q5_1 3.78%、Q5_0 4.25%、IQ4_NL 7.60%、Q4_0 8.55% です。
- ネイティブのエキスパートの Q4_0、Q4_1、Q5_0、Q6_K の gate/up の GPU カーネル。CPU のエキスパートの AVX-2 の i-quant と Q2_0 のカーネル。
- 検証の窓と prefill の経路で起動の数を減らすカーネルの融合（#783 ほか）。
- `XESTRATA_PREFILL_CPU_SHARE`（既定は `auto` でプロンプトごとに測ります。0 ですべて GPU）: 短いチャンクで、ほとんど選ばれないエキスパートを空いている CPU のプールで計算します。
- `--kv-grow`: K/V は要求が届いたセルの分だけ VRAM を取り、残りをエキスパートのキャッシュが使います。
  デバイスの仮想メモリの対応づけの粒度が 2 MiB 以上なら既定で有効です（RTX 4070 は有効、B70 は 64 KiB で無効）。
- `--adapt-async`（常駐モードの既定）、`--batch-mtp`（`--batch` と `--mtp` のときの既定）、`--pipeline-windows`（opt-in、[MULTIGPU](MULTIGPU.ja.md)）、`--lookup-chain`（opt-in。HIP の統合 GPU では 2026-11-16 から 2 が既定、[890M の decode の見直し](#890m-の-decode-の見直し)）。
- setup の追加（`--inspect`、`--source modelscope`、200K の文脈など）とサーバーの変更。

速くするための経路は B70 か RTX 4070 で速くなり、もう一方で遅くならないものを移しました。
片方だけで速いものは、機器が報告する値で実行時に選ぶか、opt-in にしています。

取り込みは free（intel/llvm）、contrib-icpx（icpx）、contrib-llvm（intel/llvm と CUDA）でビルドし、どれも CTest の 71 件が通ります（`expert_multi_test` は AVX-512 のない CPU で飛ばします）。
AVX-512 の経路は Intel SDE（`-icx`）で `expert_multi_test` が通ります。
`expert_parity --selftest` と `pool_test --selftest` は正準形の pack がないので SDE では動かせません（AVX2 では通ります）。
B70 では icpx と free のビルドが 8 つの答え（思考あり・なし）をすべて最後まで出し、6 つが一致しました。
prefill 1,148 tok/s、その後の decode 98.6 tok/s、短い会話 77.3 tok/s（icpx）です。
8 GiB・4 GiB・XMX なし・CPU のワーカー 6 つの構成と、RTX 4070 でも答えを最後まで出しました。
UHD 770 では free と icpx のビルドが起動し、短い答えを返しました。

## Strata 0.1.38 の取り込み

upstream の Strata 0.1.38（`99f3dbd0b21d1401b3769e0c0d963913607f380b`）までの変更のうち、単一の GPU と Linux に関わるものを Xe に移しています。
upstream の履歴はマージとして残しています。
CUDA、HIP、Windows の実装は移していません。複数の GPU の実装は層の分割だけを移しています（[MULTIGPU](MULTIGPU.ja.md)）。

移したものは次のとおりです。

- `--coupled-draft`: MTP の下書きを対象モデルと同じサンプリングで引きます（既定は無効、`--no-coupled-draft` で明示的に無効）。
  下書きの語彙の対応表とペナルティの履歴も含みます。
  サーバーの設定では `coupled_draft` で指定でき、エンジンの引数があればそちらが優先されます。
- ファイルから読むエキスパートの層の `--resident-budget-gib`、`--resident-experts`、`--resident-cpu-experts`。
- `--shared-expert-arena PATH`: 完成したエキスパートの重みを Linux のプロセスの間で共有します。
  データを書き出してから、ファイルのロックの下で完成を公開します。
  読む側は pack の同一性を確かめてから使います。
- ネイティブの PLE キーは Q2_0、IQ3_XXS、IQ4_XS、Q8_0 を受け付けます。
  BF16 の PLE キーが pack にあれば、そちらを使います。
- ネイティブの RoPE は要求のスケーリングの設定に従います。
- MTP のダウンロードは固定したチェックポイントかどうかを確かめ、不正な範囲を拒みます。

速くするための経路は B70 で測って速くならなかったので移していません。

| 経路 | B70 での結果 |
| --- | --- |
| prefill の量子化したエキスパートの積（`XESTRATA_PREFILL_MMQ`） | 既定の XMX の FP16 の経路の 0.34 倍 |
| 融合 MoE（`XESTRATA_PF_FUSED`） | 既定の経路の 0.25 倍 |
| GDN の key-head の共有（`XESTRATA_GDN_KEYHEAD`） | 既定の経路の 0.54 倍 |
| hyper-connection の分割読み出し（`XESTRATA_GR_V3`） | decode が約 18% 遅い |
| CPU のプールの P コアと E コアの配置（`--pool-affinity`） | decode が 1〜4% 遅い |
| AVX2 の IQ カーネルのソフトウェアのプリフェッチ | 差はばらつきの範囲 |
| ファイルの層のルーティングによる先読み（`XESTRATA_LOOKAHEAD`） | decode が約 2% 遅い |

量子化した積と融合 MoE は XMX を使わずに decode と同じ DP4a の内積で計算します。
そのため、XMX のある GPU では既定の経路に勝てません。

取り込みは free（dpclang++）と nonfree（icpx）の両方でビルドし、どちらも CTest の 52 件が通ります。
既定の構成（IQ2_XS、nonfree）の出力は、取り込み前（`bae99bd`）と比べました。
18 トークンのチャットから 32 トークンを生成すると、2 回とも一致しました。
4,095 トークンのプロンプトから 16 トークンを生成すると、取り込み前は 3 回とも同じでした。
取り込み後は 3 回のうち 2 回が一致し、1 回は最初のトークンから違いました（[実行ごとに出力が変わる理由](#実行ごとに出力が変わる理由)）。
AVX-512 の経路は Intel SDE（`-icx`）で、`expert_multi_test` と `native_expert_parity --synthetic`（iq3_xxs/iq4_nl、q4_K/q5_1）が通ります。
`expert_parity` と `pool_test` は正準形の pack がないので飛ばしました。
統合のコミットでは 8 GiB の制限、XMX なし、CPU のワーカー 2 つの構成で、`--coupled-draft` の有無で同じ 16 トークンが出ることを確かめました。
UHD 770 では free のビルドが 2 トークンの MTP の窓で答えました。
nonfree のビルドの 4 トークンの窓は Intel のランタイムの中で止まりました。

## 実行時の約束

エンジンが GPU とホストに求めることと、守っている約束です。

- BIOS: 単体の GPU では Above 4G Decoding と Re-Size BAR を有効にし、CSM を無効にします。
  そうしないと B70 の BAR が割り当てられず、xe ドライバーがつながりませんでした。
- GPU の選び方: GPU を 1 枚使います（層の分割では複数、[MULTIGPU](MULTIGPU.ja.md#gpu-の番号)）。Intel の GPU は Level Zero、NVIDIA の GPU は CUDA のバックエンドを通します（OpenCL は同じ Intel の GPU を重ねて見せるので使いません）。
  メーカー名やデバイス ID では選ばず、報告される能力で選びます。
  32 幅のサブグループ、FP16、デバイスとホストの USM が必要です。
  実行ファイルがその GPU 向けのコードを持たないとき（`STRATA_CUDA_ARCHS` なしのビルドと NVIDIA の GPU など）も断ります。
  `XESTRATA_GPU_PCI`（setup が書きます）は GPU を PCI アドレスで指定します。
  これがなければ、単体の GPU を CPU 内蔵のグラフィックス（Level Zero の integrated フラグ。他のバックエンドの GPU は単体とみなします）より先に選び、
  その中ではメモリの多いもの、同じならば計算ユニットの多いものを選びます（計算ユニットの大きさはメーカーで違うためです）。
  CPU への代替の経路はありません。
  必要なものがない GPU は理由を示して断ります。
- キュー: 1 つのコンテキストが順序つきの計算キューと転送キューを 1 つずつ持ちます。
  `copy_async` はイベントを返し、`compute_after` がキューをまたぐ依存を入れます。
  読み戻すときは、計算の完了イベントを `copy_async` に渡し、待ってからホストのバッファーを読みます。
  既存の parity のテストはホスト USM の別の確保を中継に使ってこの経路を通しています。
  `create_stream` は同じコンテキストに順序つきのキューを足します。CUDA が別のストリームに置いていた作業のためです。
  `destroy_stream` は 1 つを空にして解放し、`finish()` はすべてを空にします。
  カーネルが受け取る `void* stream` は `Runtime::stream` が解決します。
  エンジンのストリーム、計算キュー、または null（計算キュー。CUDA が同期していたところでは同期して完了）のどれかで、それ以外のポインターは断ります。
  別のストリームの作業の順序は CUDA の非ブロッキングのストリームと同じく、イベントでだけ決まります。
- 確保: デバイスの確保は 4096 バイト境界にそろえ、バンプアロケーターは容量を確かめ、折り返しません。
- エキスパートの置き場（アリーナ）: `PinnedArena` は 2 MB 境界にそろえた匿名のマッピング 1 つで、`MADV_HUGEPAGE` を付けます。
  ローダーが中身を詰める前に `prepare_for_device_copy` でデバイスへのコピー用に登録し、マッピングや登録に失敗したら例外を投げます。
  ホストの USM は使いません。
  Level Zero は `max_mem_alloc_size`（B70 で 32.5 GB）を超える 1 つの USM の確保を断り、Coder 以外のどのモデルのアリーナもそれより大きいためです。
  登録は、デバイスの最大の確保量を超えない、層の始まりで切れる断片に分けます。
  UHD 770（4 GiB）は、それより大きい登録をエラーなしで受け付け、間違ったバイトをコピーしました。
  `registered_bytes` は容量全体です。hugetlbfs のプールは使わず、透過的なヒュージページで支えます。
  登録に memlock の上限は要りません（B70 で、上限 8 MiB のまま 47.46 GiB のアリーナを登録できました）。
  Level Zero の登録（xe や i915 の userptr）はページを固定せずに追いかけるので、メモリの compaction がページを 1 つ動かすと、ドライバーは次のコピーで登録全体を GPU に写し直します（#42）。
  そのため詰め終えたアリーナは、`vm.compact_unevictable_allowed` が 0 なら `mlock` し（`locked_bytes`）、そうでなければ io_uring の固定バッファで固定します。
  共有のアリーナ（memfd）は固定しても動かされるので、`mlock` と `vm.compact_unevictable_allowed=0` でだけ守れます。どちらも memlock の上限がアリーナの分要ります。
- **デバイスの能力に合わせる**: カーネルはデバイスの報告に従います。
  1024 個のワークアイテムのワークグループは、それを受け付けるデバイスでだけ使います（UHD 770 は 512 まで）。
  FP64 はそれを持つデバイスでだけ使い、持たないデバイス（UHD 770 と Arc の A シリーズ）では FP32 で足し合わせます。
- 検証の窓での待ち合わせ: 検証の窓（verify window）の GPU は、ホストのメモリのフラグを回し読みして CPU のエキスパートを待ちます。
  これには、動いているカーネルからホストの書き込みが見える必要があります。
  検証器は起動時にそれを確かめ（`doorbell_visible`）、見えない GPU（UHD 770）では待つカーネルを使わず、ホストが区切りを順に起動します。
  結果は同じで、遅くなります。
  `XESTRATA_VERIFY_SEGMENTED=1` と `0` で、どちらかに固定できます。
  検証器の層のエキスパートがすべて VRAM にある窓は、エキスパートの割り当てを GPU が自分で決める別のグラフで動かし、ホストを待ちません（upstream cfd3b72。区切りも要りません）。
  B70 の Coder IQ1_M では 28.0 から 28.5 tok/s になり、logits は同じでした。
  `XESTRATA_VERIFY_RESIDENT_GRAPH=0` で使わないようにできます。
- **寿命**: コピー元、コピー先、読み手は、最後の使い手が終わるまで生かしておきます。
  コピーの完了イベントだけでは、カーネルがまだコピー先を読んでいる間の再利用は許されません。
  アリーナを壊すときは、USM を解放したり登録を解いたりする前に、持っているキューをすべて空にします。
  USM のポインターは確保したコンテキストで使い、その依存はアプリケーションが管理します
  （[Khronos の USM の説明](https://github.com/KhronosGroup/SYCL_Reference/blob/main/source/iface/usm_basic_concept.rst)）。
- エラーと待ち: SYCL の同期的な失敗は CLI まで伝えます。
  非同期のエラーは保持して、待つときに確かめます。
  待ちはドライバーの中でブロックし、見張りのスレッドが 120 秒続いた待ちをその名前とともに報告してプロセスを終えます。
  xe ドライバーは `job_timeout_ms`（ここでは 5 秒）でジョブをリセットするので、正常な待ちはその時間より十分早く戻ります。
  ハングや、安全に後始末できない失敗では、生きている GPU の確保の上を巻き戻さずにプロセスを終えます。
  ドライバーのジョブのタイムアウトは変えず、CTest は外側に 60 秒のタイムアウトを置きます。

parity のテストは埋め込みとスケールの小さな記録と再生の確認を、実験的な [oneAPI の graph 拡張](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_oneapi_graph.asciidoc)で続けています。
これは、トークン全体のグラフや CPU と GPU のドアベルの手順を確かめるものではありません。

## 移植した計算

### 要素ごとの計算

Xe のライブラリは、埋め込みの行の取り出し `embed_rows`（`iq_kernels.hpp`）、`scale_inplace`、`f32_to_f16_bulk` を提供します。
`elementwise.hpp` のほかの宣言は移植しておらず、ライブラリにはありません。
これらのカーネルと parity の参照は高速な数学関数と積和の融合（contraction）を切ってコンパイルします。

デバイスのコードはすべて `-foffload-fp32-prec-div -foffload-fp32-prec-sqrt` でビルドします。
CUDA は既定で除算と平方根を IEEE の丸めで行いますが、SYCL のデバイスのコードは行いません。
`quantize_act_parity` は Q8_K の `1/iscale` に出る 1 ulp の差でこれを見つけます。

| 計算 | 参照と、変えていない合格の基準 |
| --- | --- |
| 圧縮された埋め込みの行 | 独立したスカラーの展開、乗算と加算を別々に丸める、ビット単位の一致、符号つきの 0、ガード。108 の行のケースと、記録した実行 |
| GDN のゲート | CPU の double の softplus（`x > 20` の分岐つき）。相対 L1 が `1e-6` 以下、負のゲート |
| SiLU | CPU の double の式。相対 L1 が `1e-7` 以下。Xe は安定な FP32 の式を使う |
| スケーリング | CPU の FP32 の乗算。ビット単位の一致 |
| FP32 から FP16 | 共通の整数の変換。ビット単位の一致 |
| RMS ノルム | CPU の FP64 の和と平均、イプシロン、任意の重み。相対 L1 が `1e-6` 未満、有限の出力、元の null の重みの確認 |

テストのケースは足さず、入力、CPU の期待値、許容誤差は元のままです。
NumPy で求めた FP16 の境界の確認も含む `quantize_act_parity` は、B70 で通ります。
ケースを足すには承認が要ります。

### i-quant と MMVQ

`iq_kernels.hpp` はすべて移植しました（`src/kernels/xe/iq_kernels.cpp`）。
q8_1 の量子化、IQ1_M、IQ2_XXS、IQ2_XS、IQ2_S、IQ3_XXS、IQ3_S、IQ4_NL、IQ4_XS、Q3_K、Q2_0 の展開、その MMVQ、
埋め込みの行、prefill の経路の gate と up の展開、まとめて計算するネイティブのエキスパートです。
ワープは 32 幅のサブグループにし、その和は CUDA のバタフライの順に取り、CUDA のバイトの組み込み関数は書き下しました。
まとめて計算するエキスパートの SwiGLU は、CUDA の `__expf` の代わりに精度の高い指数関数を使います。

`native_mmvq.hpp` もすべて移植しました（`src/kernels/xe/native_mmvq.cpp`）。
Q2_0、Q3_K、Q4_K、Q5_K、Q6_K、Q4_0、Q5_0、Q8_0、IQ4_NL、IQ4_XS の llama.cpp の MMVQ のアダプター（1〜8 列、正確な配置と upstream の配置）と、振り分けです。
振り分けは CUDA と同じく、ほかの i-quant を `iq_mmvq` に回します。
1 列の場合も複数列のカーネルを CUDA の 1 列の配置で使い、別の 1 列用のカーネルは置きません。
CUDA のファイルでも、両者はビット単位で同じに保たれていました。
呼び出しには明示的なランタイムのストリームが要り、待たずに投入します。
CUDA の約束が null でないストリームを求めていたのと同じです。

i-quant の測定では、10 の形式すべてが実際のモデルの行を gguf-py と同じに展開しました（Q2_0 は ggml から写した定義）。
MMVQ は `iq_parity` の基準 `2e-2` の中（4.2e-3〜5.8e-3）で、`iq_parity` の失敗は 0 です。
Q6_K、Q8_0、Q4_K、Q5_K、Q4_0、Q5_0 の MMVQ は、ランダムなブロックで FP64 から `2e-2` 以内です。
2〜8 列は正確な配置で 1 列の呼び出しとビット単位で同じです。

### decode のそのほかのカーネル

- `src/kernels/xe/rope.cpp`: `rope.hpp`、`native_rope.hpp`、`mrope.hpp` の M-RoPE の表（Xe のデバイスは 1 つなので、ポインターも 1 つ）
- `src/kernels/xe/router.cpp`: `router_top10.hpp` と `native_router.hpp`
- `src/kernels/xe/bf16_gemv.cpp`: `bf16_gemv.hpp`（ggml の mmvf のカーネルを含む）
- `quantize_act.cpp`、`qsa.cpp`、`kv_q8.cpp`、`kv_q4.cpp`、`qsa_decode_attn.cpp`、`kv_stream.cpp`: 同じ名前のヘッダー。
  QSA の入り口は、CUDA が終了していたところで `DeviceError` を投げます

KV のホスト側のプール（`KvHostPools`）はカーネルが読み書きするので、Xe ではホストの USM でなければなりません。
デバイスへのコピー用に登録しただけのメモリは、カーネルから参照できません。
`native_rope` と `native_router` は CUDA が高速な数学関数を使っていたところで精度の高い `pow`、`cos`、`sin`、`exp` を使います。
decode のカーネルの確認では `rope_parity`、`router_top10_parity`、`bf16_gemv_parity` が通り、どれも CTest に登録しています。

## B70 で確かめたモデル

どのモデルも同じ手順で確かめました（手順）。

- すべての種類のエキスパートの組を含む層での `native_expert_parity` と、`dequant_bf16_test`
- 短いプロンプトと、固定したコミットの llama.cpp（CPU）に対する 96 トークンの貪欲生成の比較
- エキスパートの配置を固定した、3 つの推測の方式
- server（ストリーミング、取り消し、3 往復の会話）
- 26,293 トークンのプロンプト。KV のストリーミングのありとなし

ファイルはテンソルの一覧で固定しています。

| モデル | 結果 | llama.cpp と比べて、最初に分かれるまでのトークン数（そこでの参照の上位 2 候補の差） | 推測の方式 |
| --- | --- | --- | --- |
| Qwen3.8-Flash-Next IQ2_XS | 動く | 推測なしで 96 のうち 96 | 同じトークン |
| Qwen3.8-Flash-Next IQ3_XXS | 動く | 6（0.090） | 異なる（下を参照） |
| Qwen3.8-Flash-Next IQ3_S | 動く | 61（0.051） | 推測なし = suffix。MTP は同じ僅差のところで異なる |
| Qwen3.8-Flash-Next Q2_0 | 動く（ネイティブの pack。AVX-512 の正準形の pack はこの CPU では動かしていない） | 47（0.050） | 同じトークン |
| Qwen3.8-Flash-Next Coder IQ1_M | 動く | 20（0.238） | 同じトークン |
| Swift 1.5 IQ2_XS | 動く | 7（0.151） | 推測なし = suffix。MTP は異なる |
| Swift 1.5 IQ3_XXS | 動く | 62（0.058、3 候補） | 異なる（キャッシュの大きさも実行ごとに違った） |
| Swift 1.5 Q2_0 | 動く（注） | 25（0.273） | 推測なし = suffix。MTP は異なる |

注: Swift 1.5 Q2_0 は、層 13 のエキスパートのテンソルが 2 つのシャードにまたがります。
pack の索引はテンソルごとにシャードを記録します。
2026-10-02 より前の pack は v4、それより新しいものは XeStrata 独自の xs1 と書きます。
xs1 は upstream の v&lt;N&gt; とは別に番号を振っているので、どちらも他方と取り違えられません。

llama.cpp と分かれるのは、どれも参照の上位 2 候補が僅差のところです。
そこから先は比べても何も言えません。
ネイティブの経路はプロンプトの最後のトークンから `--dump-logits` の行を書きます（プロンプトは出力ヘッドなしでまとめて読むため）。
上の記録はその機能より前に取ったので、トークンだけを比べています。

推測の方式で結果が分かれたのは、主に次の理由です。

- MTP の実行では、推測器が VRAM を使うので、キャッシュに入るエキスパートが減ります。
  そのぶん一部のエキスパートが GPU ではなく CPU で計算されます。
  2026-10-17 からは、gate と up が i-quant（IQ2_XXS・IQ2_XS・IQ2_S・IQ3_XXS・IQ3_S）・IQ4_XS・Q2_0、down が IQ4_NL・Q2_0 のエキスパート（Qwen3.8 の 4 つのファイルのすべての層）は、CPU と GPU（Intel・NVIDIA・AMD）で同じ計算をし、どちらで計算してもビット単位で同じ結果になります（`canon_math.hpp`、`canon_expert.cpp`）。
  SwiGLU と int8 のスケールは足し算・掛け算・fma だけで作り、gate と up の活性は 256 個に 1 つのスケールで量子化し、スケールを共有する部分の整数の和を先に足してから、GPU のレーンと同じ順に fma で足します。
  窓の中身、PCIe の分担、入れ替えでエキスパートが CPU と GPU を行き来しても、答えは変わりません。
  K-quant や BF16 のエキスパート（Unsloth のファイル、Ornith、llm-jp）は対象外です。
  `XESTRATA_CANON=0` で CPU を以前のカーネルに戻します（比べるため。GPU は戻らないので、CPU と GPU の結果は一致しなくなります）。
- Qwen IQ3_XXS では、キャッシュが同じでも、suffix の実行と推測なしの実行が分かれていました。
  CPU の複数トークン用のエキスパートのカーネル（i-quant の gate と up、IQ4_NL の down）を 2 トークンから使い、1 トークンでは ggml の内積を使っていたため、トークンの行が 1 つのときと窓の中のときで足す順が違ったためです。
  2026-10-16 からは、CPU はこれらと Q4_K・Q5_K・Q6_K を 1 トークンから複数トークン用のカーネルで計算し、Q2_K・Q3_K・Q8_0 はどのトークン数でも ggml の内積で計算します。トークンの行は 1 つのときも窓の中でも同じになり、キャッシュが同じなら推測の方式によらずロジットがビット単位で同じになります（複数トークンの窓の GPU 側は、もともと差を生んでいません）。
  `XESTRATA_IQ_MT_MIN=N` で、以前のように N トークンから複数トークン用のカーネルを使います（比べるため）。
- AVX-512 の CPU では i-quant の格子の引きを 1 つの gather 命令で行えます（upstream 0.2.5 の移植）。読む語は同じなので、結果はスカラーの引きとビット単位で同じです。
  既定でオンになるのは、CPUID が AVX-VNNI と hybrid を報告する CPU の P コアだけで（`cpu_gather_fast`。ベンダー名やモデル番号では決めません）、Zen 4・Zen 5 では gather がマイクロコードで遅いので既定でオフです。`STRATA_IQ_GATHER=1` / `0` でどちらにも固定できます。
  gather がオンの AVX-512 の CPU では、IQ3_XXS と IQ3_S のエキスパートの 1 トークンの gate と up を、ggml の 1 トークンの内積とビット単位で同じ exact rows（`iq512x_rows`）で計算します。最初に使うときに ggml と照合し、合わなければ使いません。`STRATA_NO_IQ512X1=1` でオフにできます。
  canonical の行を取る層（この fork の既定）では、どちらも使いません。

### 実行ごとに出力が変わる理由

どのエキスパートを CPU で計算するかはキャッシュの大きさで決まり、それが出力の最後の数ビットを変えます。
`auto` は空いている VRAM からキャッシュの大きさを決めるので、起動ごとの VRAM の揺れでスロットが 1〜2 個動き、出力が変わることがあります。
64 スロット単位に切り捨てれば同じ機械では同じキャッシュになりますが、RTX 4070 で decode が 3〜4% 遅くなったので、速さを取って切り捨てていません。
代わりに setup のウォームアップが、起動したときに空いていた VRAM から 32 MiB 引いた値を実行の設定に書きます（`--vram-free-mib`）。エンジンは起動時にそれより多く空いていても数えないので、空きが少し多い起動でも同じキャッシュになります（B70 で起動ごとに 20 MiB ほど揺れ、スロットが 15 個動いていました）。空きがそれより少ない起動では、キャッシュは小さくなります。
同じ出力が要るとき（機械や設定をまたいだ比較も）は、`--expert-cache N` で固定します。

密な行列積の道も起動ごとに変わります。
プロンプトと検証の窓の密な行列積は、形（種類・行数の級・N・K）ごとに XeStrata のカーネルと oneMath の時間を起動時に測り、XeStrata のカーネルが 5% 以上速いときだけそちらを使います。
ライブラリの速さは実行ごとに 2 倍ほど動くことがあり（RX 9060 XT の 4096 x 320 x 10240 の BF16: rocBLAS 13.1〜25.5 TFLOPS、XeStrata のカーネル 16.8〜21.3）、選びが XeStrata とライブラリの間で動くと最後の数ビットが変わります（同じ実行物・同じ設定で 64K の greedy が 18 トークン目で分かれた）。
ライブラリの速い/遅いの 2 つの状態はビットを変えません（`XESTRATA_BLAS=1` の 2 回は最初の 48 窓まで同じハッシュ）。

検証の窓の大きさも起動ごとに変わります。
`--spec-window auto` は大きさごとに測ったラウンドの時間（実時間）から窓の大きさを選び、`--suffix-draft` は繰り返しの位置で窓を `--spec` より広げます。
窓の大きさが違うと同じ位置の行でも計算の形が変わり、最後の数ビットが変わります（XeStrata のカーネルだけでも、3 トークンの窓と 4 トークンの窓で同じ位置の argmax が違った）。

同じ答えが要る実行（機械や設定をまたいだ比較、md5 の見張り）は、次の 3 つで固定します（RX 9060 XT、64K）。

- `XESTRATA_NO_BLAS=1`: 密な行列積をすべて XeStrata のカーネルで計算する（ライブラリを測らない）。
- `--spec-window min-p --spec-min-p 0`: 窓をいつも `--spec` の大きさにする（測った時間を読まない）。
- `--suffix-draft 0`: サフィックスの下書きで窓を広げない。

この 3 つで 64K の A/A は 5 回とも同じ答えになりました（2026-11-14）。
prefill は 890〜906 トークン/秒・decode は 34.2〜34.9 トークン/秒（既定は 962〜977・37.5〜40.7）で、prefill が 6〜7%、decode が 8〜10% 遅いので、既定は速さのため測って選ぶ道のままです。

短いプロンプト（CPU がエキスパートを分担する塊）では、CPU が受け持つ割合を塊ごとに測るので、その揺れでも答えが変わります（CPU と GPU でエキスパートの行の丸めが違うため）。
`XESTRATA_EXACT=1` は、エキスパートの行を CPU でも GPU でも同じ固定小数点の整数の和で計算し、割合が動いても答えを変えません（Strix Halo の Qwen3.8 IQ3_XXS、1K の greedy が 5 回とも同じ。既定では 5 回で 4 通り）。
1K の prefill が 7%、decode が 3% ほど遅くなるので、既定ではありません。

### 出力ヘッドの VRAM の確保が断られる

起動時に、出力ヘッドを GPU に上げる確保が断られることがあります。
そのとき空いている VRAM は約 27 GiB あり、いつもアリーナをデバイスへのコピー用に登録した直後の、最初のデバイスの確保です。
同じ設定でもう一度起動すると、普通に動きます。
新しい開発機では約 120 回の起動のうち 3 回起きました。

### 小さい GPU での VRAM の予約

既定の予約は機器のメモリの 32 分の 1 で、700 MiB から 768 MiB の間です（plain のエンジンと同じ式、core::vram_room。16 GiB のカードでは下限の 700 MiB、24 GiB 以上では 768 MiB）。小さい GPU では（upstream #496）、その 700 MiB のせいで、prefill の経路に 256 トークンの塊を貸すだけのスロットがキャッシュに残らないことがあります。
そのときエンジンは、それだけ残るところまで予約を下げます。
下限は 300 MiB で、`--vram-reserve-mib` で指定した予約より下げることはありません。
それでも足りなければ起動を止め、何 MiB 足りないか、何を減らせば空くかを示します。
`XESTRATA_VRAM_LIMIT_MIB` で確かめました。
4,608 MiB（IQ2_XS、32K、MTP）では予約が 482 MiB になり、必要な 368 スロットが残りました。
4,096 MiB では 330 MiB 足りずに止まりました。

## 外れたエキスパートの CPU と PCIe への分け方

検証の窓で VRAM にないエキスパートは、CPU が計算するか、GPU が PCIe 越しに読んで計算します。
どちらに何個回すかを、起動時に測った時間で決めます（[Project Maya](https://github.com/mw00/project-maya) の CPU lane の考え方。コードは書き起こし）。
以前はリンクの速さだけから決めた割合（20 GB/s 以上で 0.55、それより遅ければ速さに比例）でした。

- 測るもの: CPU のプールがエキスパート 1 個を計算する時間（64 個を順に回し、キャッシュに収まらないようにする）、
  窓の読み込みのカーネル（`fetch_blobs`）が PCIe 越しに 1 個読む時間、窓のエキスパートのカーネルが VRAM の 1 個を計算する時間、
  CPU がエキスパートを読むメモリの帯域。
- 取り合い: CPU と PCIe は同じメモリを読みます。2 つの速さの合計がメモリの帯域を超えるとき、両方がその比で遅くなるとみなします。
  B70 では、PCIe の 1 個が単独の 0.066 ms から 0.13 ms になる見積もりで、decode から逆算した値と合いました。
  メモリの帯域は 100 ms おきに 5 回読んで最も速い値を使い、測る前にカーネルを先に用意するスレッドが終わるのを待ちます。generate では prompt を読む前に測ります。
  1 回の読みはほかの仕事と重なると 3 倍まで低く出て、RX 9060 XT では起動ごとに外れを全部 CPU に回す判定になり、64K の decode が 21 tok/s に落ちていました（そろえた後は 38〜41 tok/s）。
- 分け方: 窓のグループごとに、GPU が VRAM 上のエキスパートを計算した後（読み込みを並べられる GPU では並べて）PCIe の分を読んで計算し、
  CPU が残りを計算するとして、遅い方が最も早く終わる数を選びます。
  CPU だけの場合より 10% 以上早く終わる見積もりのときだけ PCIe に回します。
- decode の実測で詰め直す: 起動時の CPU の時間は 1 トークンずつ測るので、decode の窓のプールの呼び出しはそれより遅くなることがあります（RX 9060 XT で 1.27〜1.29 倍、ホストが忙しいときは 3.5 倍）。
  decode の呼び出しの実時間を、分け方が見込んだ時間と比べ、15% を超えてずれていれば CPU の時間をその比で直します（CPU に計算のあった呼び出しの 512 回、さらに 1,024 回、2,048 回の後、3 回まで。upstream の PcieAuto が割合を詰めるのと同じ考え方）。
  RX 9060 XT の IQ3_XXS で decode が 1K 50.3 → 52.2、64K 48.5 → 51.3、262K 42.5 → 44.7 tok/s になり、起動時の時間が実測と合う B70 と RTX 4070 では動きません。`XESTRATA_PCIE_SPLIT_REFINE=0` で起動時の時間のままにします。
- **領域**: Level Zero では、デバイスへのコピー用に登録したエキスパートの領域を GPU のカーネルが読めません。
  分け方が PCIe に回すときだけ、領域を層ごとにホスト USM へ移します（IQ3_XXS の 40 GB で約 7.5 秒）。`XESTRATA_ARENA_HOST_USM` は 16 MiB を両方の置き場から GPU へ写して比べ、ホスト USM が 2% 以上速いときだけ最初からホスト USM にします（同じなら、OS がページを取り返せる写像のまま。`0`・`1` で固定）。
- **READY の前の待ち**: エンジンは READY と言う前に、カーネルがホストのページを動かし終えるまで待ちます（`/proc/vmstat` の `pgmigrate_success` が 2 秒続けて毎秒 10,000 ページ未満、最大 30 秒。Linux だけ）。
  読み込みの直後のメモリの詰め直しは登録した領域のページも動かしうるので、それを最初の要求より前に済ませます。`XESTRATA_SETTLE_S` で最大の秒数を変え、0 で待ちません。
- **割合のまま**: `--pcie-frac`（`--calibrate` が保存した値を含む）を渡したとき、層の分割、`--pcie-mode dma` と `direct`、測れなかったときは、以前の割合を使います。

RTX 4070（PCIe 4.0 x4）では外れを全部 CPU に回し、IQ3_XXS 32K で 49.9 → 57〜58.6 tok/s、UD-Q4_K_XL 32K で 22.0 → 25.7 tok/s です。
B70 も全部 CPU（以前と同じ）で、RX 9060 XT は PCIe に回す数が 1 層 4.7 個から 1.0 個に減り、どちらも速さは変わりません。

## 画像

画像は、setup で選んだときだけ使えます（既定はオフ）。
選ぶと、画像のエンコーダーは既定で GPU で動きます（`./setup.sh --vision yes` または `gpu`）。
エンコーダーは XeStrata 自身の視覚エンコーダー（画像エンジンと共通の `src/core/vision.cpp`。モデルの mmproj の Qwen3-VL の ViT と merger）で、エンジンと一緒にビルドされる 1 つの実行ファイル（`xestrata-vision`）が、GPU（`--gpu`）でも CPU でも動かします。
GPU で動かせないときは CPU のエンコーダーに切り替わり、`--vision cpu` では CPU のエンコーダーだけを使います。
CPU のエンコーダーは、重みを読み込むときに int8 にし、活性は行ごとに量子化して、XeStrata の CPU の int8 の積（AVX2・AVX-VNNI・AVX-512 VNNI を CPU の報告で選ぶ）で計算します。
どちらのエンコーダーも、画像を llama.cpp の mtmd と同じ格子（大きさの規則と黒の余白）にします。
動くモデルはすべて画像でも確かめました。
2 つの形の格子、1 つの要求に 2 枚の画像、同じ画像をもう一度送ってそのキャッシュを使う会話です。

GPU を PCI アドレスで選びます（`xestrata-vision --gpu --gpu-pci`、設定の `vision.gpu_pci`）。
最初の GPU が CPU 内蔵のグラフィックスやほかの GPU のこともあるためです。
GPU のエンコーダーは CPU のエンコーダー（8 スレッド）より 20 倍ほど速く（1,024 トークンの絵で RX 9060 XT が 0.2〜0.3 秒、Ryzen 7 5700X が 5.3 秒）、埋め込みは毎回ビット単位で同じになります。
CPU のエンコーダーは、以前の CPU のエンコーダー（llama.cpp の mtmd）の約 2 倍速く（同じ 5700X・8 スレッドで 1 枚 5.3 秒 対 mtmd 10.2〜11.2 秒）、300 トークンの絵で 1.1 秒です。埋め込みは mtmd のものと画像のトークンごとのコサインが平均 0.9964 です。
正答は 300 トークンで 18/36、1,024 トークンで 28/36 です（BF16 では 19/36・26/36）。
環境変数 `XESTRATA_VISION_CPU_I8=0` を置くと、CPU のエンコーダーは積を BF16（重みも、活性は FP32 のまま）にし、mtmd とのコサインは平均 0.9998 に上がりますが、約 2 倍遅くなります（同じ 5700X で 300 トークン 2.5 秒・1,024 トークン 9.4 秒）。
（contrib-icpx のモードにあった SYCL のビルド（ggml-sycl、oneMKL）は、そのモードと一緒に 2026-10-15 に廃止しました。）

server はどちらかが準備できるまで、エンコーダーを次の順に起動します。

1. GPU のエンコーダー
1. CPU のエンコーダー（設定の `vision.fallback`）

どれが使われ、それより前のものがなぜ起動しなかったかは、コンソールと `/v1/status`（`vision.encoder`、`vision.fallback`）に出ます。
どれも起動しないとき、または使っていたエンコーダーが後で止まったときも、server は文章の応答を続け、`vision.error` に理由を書きます。

## IQ2_XS のテンソルの構成

モデルの一覧は Hugging Face のリビジョンと、各 GGUF のヘッダー全体の SHA-256 を記録しています。
テンソルの CSV は各テンソルの実際の型、形、オフセット、バイト数を記録しています。
この一覧のために取ってきたのはヘッダーだけで、ヘッダーの先頭部分はモデルのシャードとしては使えません。
モデル全体での生成、prefill の経路、MTP とプロンプト照合の推測、server、KV のストリーミングつきの 26k トークンのプロンプトも、IQ2_XS で確かめました。
貪欲生成のトークンは CPU の llama.cpp と比べました。

記録したリビジョンの `IQ2_XS` の配布物には、12 の格納形式で 1,224 のテンソルがあり、**IQ2_XS で格納したテンソルは 1 つもありません**。
この名前は精度の混ざったモデル全体の量子化の目安を表しています。

| テンソルの役割 | 実際の格納形式 | 必要な計算と参照 |
| --- | --- | --- |
| ルーティングされるエキスパートの gate と up | 34 層で IQ2_S、11 層で IQ2_XXS、3 層で IQ1_M | 形式ごとの展開、活性化の量子化、選んだ行の GEMV とまとめた GEMM。固定した ggml の `to_float` と CPU の FP64 |
| ルーティングされるエキスパートの down | 48 層すべてで Q2_0 | ネイティブの Q2_0 の展開と内積、CPU の AVX2 の経路。同じ ggml のエキスパートの参照 |
| 量子化された密な重みと共有エキスパート | IQ3_S、IQ4_XS、IQ4_NL、Q6_K、Q8_0、Q2_0 | 型ごとの展開、密な GEMV と GEMM、SwiGLU。固定した ggml と FP64 の累積 |
| トークンの埋め込みと出力ヘッド | IQ4_XS | 行の展開、射影、サンプリング。固定した ggml の展開とホストの計算 |
| HC、ルーター、SSM、インデクサー、PLE の射影 | BF16 | BF16 の活性化の丸めと射影。既存の BF16、GR、ルーター、GDN、QSA の parity |
| ノルム、バイアス、再帰のパラメーター | F32。PLE の畳み込みは F16 | RMS、ゲート、畳み込み、再帰、索引。既存のホストの参照 |
| PLE の n-gram の表（2 つ目のシャード） | IQ4_NL | 整数の n-gram のハッシュ、選んだ行の読み出しと展開。既存の PLE のベクトルと実際の表の行 |

`src/kernels/native_expert_parity.cpp` のネイティブのエキスパートの参照は、ggml の展開とホストの FP64 の内積を使ってから、CPU と GPU の経路を比べます。
上の型の gate と up、down の層では、CPU の行と GPU の行がビット単位で同じかも、トークンをまとめたときと 1 つずつのときで確かめます。
エキスパートごとの相対 L1 の基準 `3e-2` は、この比較のためのもので、カーネル全般の許容誤差ではありません。
GPU の q8_1 の活性化の約束と、CPU の活性化（上の型のエキスパートは `canon_expert.cpp` の形、それ以外は ggml の `vec_dot_type`）は、明示したまま保ちます。

## 検証の一覧

| テストと資産 | 状態と限界 |
| --- | --- |
| `xestrata-device --selftest`、`elementwise_parity` | 移植済み。B70 で通る。デバイスの自己診断は確保、境界、容量を確かめ、数値の読み戻しは確かめない |
| `gguf_reader_test`、`suffix_drafter_test`、`controller_test`、`draft_policy_test`、`conv_cache_test` | 今のツールチェーンで通る。エンジン全体の結合を示すものではない |
| `s_gemv_parity`、`s_gemv_q8k_parity` | 移植済み。CTest に登録。B70 で通る。`--bench` は `s_gemv_split`、`s2_gemv_quads`、`s2_gemv_fast` も素朴なカーネルと比べる |
| `dequant_s2_parity`、`s2_gemv_parity`、`s2_gemv_q8_parity` | 移植済み。CTest に登録。B70 で通る |
| `rope_parity`、`router_top10_parity`、`bf16_gemv_parity`、`quantize_act_parity`、`qsa_parity`、`kv_q8_parity`、`kv_q4_parity`、`kv_stream_parity`、`gdn_parity`、`gr_parity`、`cvec_parity` | 移植済み。CTest に登録。B70 で通る |
| `shared_expert_parity` | 移植済み。CTest に登録。B70 で通る（ネイティブのスカラーのゲートの SYCL graph での再生を含む） |
| `sampler_parity` | 移植済み。CTest に登録。B70 で通る |
| `qsa_prompt_attn_parity` | 既存の FP64 の参照と FP32 の基準を SYCL に移植（ベンチマークも兼ねるので、ビルドするが CTest には登録しない）。XMX のカーネルは int8 と FP16 の KV で通る。Tensor Core の版は INT8・FP16・K8V4 で通り、Q4_0 は飛ばす。行列エンジンがなければ、呼び出し側は `qsa_decode_attn_batch` を使い続ける |
| `native_moe`、`native_gdn`、`native_gdn_preprocess`、`native_ple_postops`、`native_qsa`、`native_qsa_score`、`native_qsa_indexer`、`native_flash_attn`、`qsa_select`、`fused_gdn`（ツリーに parity なし） | 固定したコミットの ggml-cpu に対するプローブが B70 で通る。CTest には登録しない |
| `native_expert_parity`、`dequant_bf16_test` | 移植済み。`STRATA_NATIVE_EXPERTS` でビルド。モデルのシャードが要るので登録しない。[B70 で確かめたモデル](#b70-で確かめたモデル)のすべてのシャードで通る |
| `iq_parity` | 移植済みでビルドする。登録しない。`tools/iq_fixture.py` が、固定したモデルのリビジョンの範囲読みから `logs/iq_fixture` を書く。10 の形式すべてが通る。Q6_K と Q8_0 は一覧にない |
| `ple_parity` | `ple_oracle_vectors.inc` をリポジトリに持つ。全体の確認には実際の表と参照の資産も要る |
| CPU のエキスパートとプールのテスト、`pool_stress` | この一覧には含めていない。正準形のエキスパートのテストには AVX-512 か pack、またはその両方が要る。AVX-512 の経路は AGENTS.md のとおり Intel SDE で動かす |
| `serve.test_server`、`serve.test_detok`、`serve.test_mcp` | 模擬のエンジンで通る |
| `tools/test_iq_pack.py`、`tools/test_shards.py`、`tools/test_calibrate.py` | `python -m unittest` で通る（2026-10-04） |
| `src/ngram/ple_reader_test.cpp`、`src/platform/memory_test.cpp` | ビルドし直した。`ple_reader_selftest` は通り、`platform_memory_test` は memlock の上限 8 MiB では失敗する。どちらも登録したアリーナの証拠にはならない |

## Xe で測り直していない upstream の設定

次の設定は upstream の Strata が CUDA（多くは Ryzen 5 7600 と RTX 5070）で決めたものです。
XeStrata はそのまま使っていて、Xe での調整は一部しか済んでいません。

| 場所 | 設定 | Xe での状態 |
| --- | --- | --- |
| `src/program/generate.cpp` | ネイティブの pack で、キャッシュにないエキスパートを PCIe で運ぶ割合 0.55（正準形は 0.2）。リンクの速さ（256 MiB の転送 4 回の最良）/ 20 GB/s で縮める | 新しい開発機で測ると、0.15 で 2〜4% 速く、それより大きい割合はどれも遅かった。0.55 はこの測定に支持されない |
| `tools/calibrate.py` | 基準は Ryzen 5 7600 と RTX 5070。PCIe の割合の候補 0、0.2、0.35、0.55、0.75、推測の下限 0.3、0.5、0.7、ワーカー数の候補。交互の確認で 3% を超えて速いときだけ採用 | 候補と基準は upstream のまま |
| `setup.py` | `--spec 4`、`--spec-min-p 0.5`、`--expert-cache auto`、`--prefill auto` | `--prefill auto` は 32768 トークンの塊まで選ぶように変えた。既定のリングで収まらない塊は、リングを 96 スロットにして選ぶ（upstream #583。B70 を 8 GB に絞った IQ3_S で、8K と 32K のプロンプトが 1.5〜2 倍）。それでも収まらなければ、int8 の積の組の作業領域（GU・H・Hq）を塊全体でなく 1 組の行数で確保して選び直す（RTX 4070 の IQ3_XXS 32K で塊が 8192 → 16384、prefill 1,194 → 2,058 tok/s。組の形で丸めが変わるので答えは上限なしの実行と少し違いうる）。窓の大きさは `--spec-min-p` の下限でなく、測った窓の時間と補正した確率から 1 ms あたりの出力で決める（`--spec-window auto`、[DETAILS](DETAILS.ja.md#この-pc-に合わせて調整する--calibrate)）。ほかは upstream のまま |
| `src/program/generate.cpp` | VRAM の予約（機器のメモリの 32 分の 1、700〜768 MiB。小さい GPU では 300 まで）、prefill の経路のスロットの借り方と塊の選び方、4 ラウンドごとのキャッシュの入れ替え（最大 96 個） | 予約の大きさを機器の報告から決めるようにした（upstream は 700 MiB 固定）。ほかは upstream のまま |
| `src/program/generate.cpp` と CPU のプール | 物理コア数に基づくワーカー数とホストのワーカー、キャッシュのヒットの通知、グラフとドアベルの段取り | ワーカー数は新しい開発機で測り、7〜19 で速さが変わらなかったので既定のまま |
| `src/kernels/cpu/iq_avx512.cpp` | ソフトウェアのプリフェッチの距離と、AVX-512 向けの調整。IQ4_XS の gate/up の 512-bit の行（upstream 55dc17697、既定で使う。`XESTRATA_IQ512_IQ4XS=0` で切る） | 開発機は AVX-512 を持たないので測っていない。IQ4_XS の行は llvm-mca の見積りで 256 値の塊あたり 1 トークン 35.3 サイクル（ggml の内積 31.8、−11%）、2 トークン 41.0（63.6 に対して +35%）、4 トークン 54.3（127.2 に対して +57%） |
