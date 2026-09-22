# Windows HIP: FastGS native atomics and SSIM, September 2026

2026-09-22、非同期処理・WMMA・rocJPEGを変更せず、CUDA由来のFastGS勾配集約と
SSIM後方計算をWindows HIP / gfx1151向けに最適化した。

## 基準と測定条件

- 基準: `f5510712693f76f3c4556929290354a28f044797`。前回の非同期・WMMA最適化を含む。
- 実装: `2dc8f6d60ae67a255682df802a47762205bc7b96`。
- Windows 11、Ryzen AI MAX+ 395、Radeon 8060S / gfx1151。
- AMDドライバ `32.0.31041.1004`、ROCm SDK `10.2.0a20260918`、HIP `7.16.26373`。
- VS 18 Insiders、Release、32並列ビルド。
- Tanks and Temples truck、251枚、979×546 / 489×273。縮小はCPU/GPUともLanczos2。
- MRNF、SH3、2,000 iterations。固定仕事量では100,000 splats、densification無効。
- 最初にexe・DLL・リソースを独立コピー。比較中にDLLの置換、別のGPU処理、ビルドを行わない。
- 両版に300 iterationの外部ウォームアップを行い、A/BとB/Aを交互に実行する。
- 通常の学習時間と診断用GPUイベント計測を分ける。プロファイルの値を最終速度比較には使わない。
- 全比較で同じrocJPEG/HIP DLLを使用。`scripts/benchmark_hip_async.ps1`にハッシュ、引数、
  loss、decode回数、評価CSVを記録する。

基準exe SHA-256: `8D1DDC5AA7C3AE6575571A8D03102F63534D668729374DE544BDBE179738E16A`。
基準lfs_core.dll: `F5A3CD9945CC1C8116D1BA56D52F19BB04AC7A6E158DA397A5EB4F47BEF66192`。
最終exe: `56C770EA8BC3E21E7075335887F6399E27A243B7C72D46D3E899CFFA40441998`。
最終lfs_core.dll: `BB739CEA0E58EFB5A3D4B0511DBCD25544AD22109A4AAA2370E558B2E0F37873`。

## truck: 最終版の通常学習比較

100,000 splats、2,000 iterations、4組の交互実行。GPUプロファイル無効。

| 条件 | 基準の中央値 | 最終版の中央値 | 時間短縮 |
|---|---:|---:|---:|
| 原寸 | 15.1845 s | 11.8850 s | 21.73% |
| 1/2、Lanczos2 | 16.2610 s | 11.3630 s | 30.12% |

| 条件 | 基準、実行1〜4 (s) | 最終版、実行1〜4 (s) |
|---|---|---|
| 原寸 | 15.112 / 15.159 / 15.210 / 15.299 | 11.655 / 11.803 / 11.967 / 12.004 |
| 1/2 | 16.380 / 16.229 / 16.293 / 16.180 | 11.363 / 11.363 / 11.356 / 11.397 |

全8組で最終版が短かった。単一画像の最終lossは画質同等性の判定には使わず、
densificationを有効にした別のholdout評価で確認する。
プロセス全体の時間中央値も原寸18.068→14.776 s、1/2は19.150→14.276 sへ短縮した。
最終版の全実行でhardware decodeは2,002回、CPU decodeは0だった。

## densificationとholdout評価

MRNF、最大300,000 Gaussian、2,000 iterations、500 iterationから100毎にdensification。
8枚ごとにholdoutへ回し、32枚を2,000 iteration時に評価した。原寸・1/2各2組、交互実行。
全runで300,000 Gaussianに到達した。この時間は通常の学習とその評価を含む同条件の比較であり、
固定100,000 Gaussianの表とは仕事量が異なる。

| 条件 | 基準の中央値 | 最終版の中央値 | 時間短縮 |
|---|---:|---:|---:|
| 原寸 | 36.5020 s | 31.0975 s | 14.81% |
| 1/2 | 28.8675 s | 21.4490 s | 25.70% |

各2回の評価指標の平均（左が基準、右が最終版）:

| 条件 | PSNR (dB) | SSIM | LPIPS |
|---|---:|---:|---:|
| 原寸 | 22.763959 → 22.762188 | 0.810974 → 0.810922 | 0.259296 → 0.259434 |
| 1/2 | 23.811136 → 23.800111 | 0.875574 → 0.875437 | 0.143242 → 0.143130 |

PSNRの平均差は原寸−0.001771 dB、1/2で−0.011025 dB。
SSIMの差は−0.000052 / −0.000137、LPIPSの差は+0.000138 / −0.000112だった。
原寸の基準自体の2回のPSNR差は0.017606 dB。bitwise一致や全データセットでの品質同等を
主張せず、この条件での小さな指標差と独立した勾配テストを検証結果とする。

## 変更の根拠

初期プロファイルでは、原寸の定常1 iteration 7.52 msに対し、backwardのGPU spanが4.55 ms、
forwardが1.24 ms、lossが0.53 msだった。CPU側の待ち時間は別指標であり、GPU spanと加算しない。

### FastGS: GPU専用勾配のネイティブFP32加算

HIPの通常のfloat `atomicAdd`がgfx1151でCASループに展開されていた。
ネイティブFP32 atomicはキャッシュ可能なcoarse-grained GPUメモリに限定する必要があるため、
コンパイラ全体にunsafe atomicsを許可する代わりに、勾配用helperのみに`unsafeAtomicAdd`を使用する。

`RasterizerMemoryArena::owns_device_allocation`は、自身のmalloc確保または自身が作ったVMMの
commit済み範囲を検証する。VMMは`PINNED + LOCATION_DEVICE`、flags=0で確保する。
ここでPINNEDは非移動のGPU確保を意味し、pinned host memoryではない。
外部import、managed、host、所有していない範囲には許可しない。
ポインタの寿命とstream依存関係は既存のForwardContext/arenaの契約を維持する。

適用先はmean2d、conic、depth、opacity、color、optional normalのhelperのみ。
camera、densification、edgeの外部出力は従来のatomicを維持する。
コンパイル時と実行時にgfx11の対応型番を確認し、それ以外とCUDAは従来経路になる。
`LFS_DISABLE_HIP_FASTGS_NATIVE_ATOMICS=1`で従来経路へ戻せる。
FP32加算順序の差やsubnormalの扱いにより、学習結果のbitwise一致は要求しない。

生成済みgfx1151コードも確認した。MRNF・法線なしのblend backwardは、従来版の
CAS命令13箇所に対し、native版は`global_atomic_add_f32` 10箇所とCAS 3箇所だった。
従来経路を残すため、object全体の命令総数だけでは比較しない。
最終exeに`LFS_DISABLE_HIP_FASTGS_NATIVE_ATOMICS=1`を設定した対照実行では、
原寸の基準15.415 sに対し15.236 sとなり、大きな短縮が消えることを確認した。
この対照は1組だけなので、残る差をSSIMによる学習全体の確定短縮率とは扱わない。

一次資料:
[AMD atomics support](https://rocmdocs.amd.com/en/latest/reference/gpu-atomics-operation.html)、
[HIP language extensions](https://rocm.docs.amd.com/projects/HIP/en/develop/how-to/hip_cpp_language_extensions.html)、
[HIP VMM implementation](https://github.com/ROCm/rocm-systems/blob/develop/projects/clr/hipamd/src/hip_vm.cpp)。

atomicのみの診断比較（各1組、GPUイベント計測あり）:

| 条件 | 基準学習 | atomic版 | backward GPU span、基準→変更後 |
|---|---:|---:|---:|
| 原寸 | 15.545 s | 12.166 s | 4.555 → 2.873 ms |
| 1/2 | 16.710 s | 12.267 s | 5.731 → 3.344 ms |

### SSIM: wave32に合わせた後方計算

HIPのfused L1+SSIM後方を32×8 tile、channelごとのblock、成分別LDS配置にした。
horizontal convolutionはhaloを含む全18行を計算する。
forwardのpartial生成、masked経路、CUDAのkernel、精度と損失式は維持する。
decoupledのmean-only経路はLDSを1成分分だけ確保する。
grid.y/zのportable limitを超える場合は従来kernelへ戻す。

GPUイベントによる後方単体測定。50回ウォームアップ後、250回×9標本の中央値。
通常のCTestではこの明示実行専用benchmarkを実行しない。

| 画像 | Target | 基準 | HIP tile版 | 時間短縮 |
|---|---|---:|---:|---:|
| 979×546 | float32 | 194.454 µs | 162.906 µs | 16.23% |
| 979×546 | uint8 | 195.604 µs | 161.157 µs | 17.61% |
| 489×273 | float32 | 52.8524 µs | 43.5456 µs | 17.61% |
| 489×273 | uint8 | 54.8808 µs | 43.2964 µs | 21.11% |

単体の短縮率を学習全体の短縮率とは扱わない。wave64、他GPU、Linuxの速度は未測定。

## 正確性・回帰確認

- SSIM 32件成功。保存済みforward partialからCPUで独立した11×11のdouble畳み込みを計算し、
  端形状、C=1/3/4、batch、uint8/float、crop有無、decoupled両勾配を比較する。
  原寸・1/2と縦長画像のfallbackは未変更のunfused後方とも比較する。
- GPUメモリ50件成功。自前malloc/VMM上の正負、高競合、小さいnormal値のnative/safe加算、
  範囲境界と外部backingの拒否を含む。
- NN 33件成功、外部fixture不足1件skip。WMMA無効時11件成功。
- 非同期allocator無効時6件成功、非同期専用条件の1件skip。
- rocJPEG・Lanczos 42件成功。

## 今回採用しなかった候補

FastGSにはすでにwarp culling、2 pixels/thread、warp単位のatomic集約、短縮depth keyがある。
重複した最適化は追加しない。CUDA由来のlaunch bounds、HIP radix sort設定、SSIM前方へのloss融合は
追加調査の候補として残す。今回は実測したatomicとSSIM後方の改善に変更を絞った。

## 展示用起動

`scripts/start_truck_exhibition.bat`でGUIの`--train`を使い、truckのimport後に自動学習する。
既定はMRNF、30,000 iterations、最大100万Gaussian、Lanczos2による1/2画像。
毎回別の出力先を作り、学習完了後もGUIを残す。詳細は[展示手順](truck_exhibition.md)。

既定設定の実起動で、Windowsのmain windowとMCPの`training.main`を確認した。
truck 251枚のimport成功後、3,605→4,975→14,385 iterationへ自動進行し、
Gaussian数は405,306→561,499→1,000,000に増加した。training errorはnull。
`lichtfeld://render/window`の画像でもtruckと学習パネルを確認した。
起動ロゴは表示を維持し、その背後で学習が進行する。
検証後もこのGUIを開いたままにし、学習を継続する。

## 再実行と記録

設定済みのROCm buildで、`BUILD_GPU_LOSS_TESTS` / `BUILD_GPU_MEMORY_TESTS`をONにする。
32並列で`LichtFeld-Studio`、`lichtfeld_loss_tests`、`lichtfeld_gpu_memory_tests`をビルドする。
損失とメモリの確認は`ctest --test-dir build-rocm10 -C Release -R 'lichtfeld_(loss|gpu_memory)' -j 1 --output-on-failure`。
SSIMの明示benchmarkは`lichtfeld_loss_tests.exe --gtest_also_run_disabled_tests --gtest_filter=FusedL1SSIMTest.DISABLED_BackwardBenchmark`。

truck比較には`scripts/benchmark_hip_async.ps1`を再利用する。
基準・変更後の独立runtimeを`-BaselineRuntime` / `-CandidateRuntime`へ指定する。
通常比較は`-Repetitions 4`、品質比較は`-Repetitions 2 -Densify -MaxCap 300000 -Evaluate`。
解像度は既定の原寸と1/2の両方、iterationsは既定の2,000。

ローカルの測定記録は次の`benchmark_outputs/`配下に保存した。

- `hip_kernel_baseline_profile_20260922`: 初期profile、build/testログ、SSIM単体、native ISA要約。
- `hip_native_atomics_screen_20260922`: atomicだけを変更した診断比較。
- `hip_native_ssim_final_20260922`: 確定実装の4組比較、runtimeハッシュ、loss、decode回数。
- `hip_native_ssim_quality_20260922`: densificationとholdout CSV。
- `hip_native_atomics_fallback_20260922`: native無効の対照実行と環境変数記録。
- `hip_exhibition_20260922`: GUI起動情報、MCP状態、ウィンドウ画像。

比較用runtimeコピー2個、ISA抽出物、各比較の一時設定・学習モデルは削除した。
集計の根拠となる記録と、展示GUIの実際の学習出力は保持する。
