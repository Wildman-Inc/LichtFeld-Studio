# Windows HIP asynchronous memory and RDNA WMMA, September 2026

2026-09-22、rocJPEGを変更せずにLFSのHIPメモリ管理とFP16 GEMMを最適化した。
最初に現行コードの実行ファイル・DLL・リソースを独立したディレクトリへ保存し、
以後はその実行環境と変更後を交互に測定した。
確定コミットのtruck学習は等倍3.75%、1/2で4.45%短縮した。
MoGe相当形状のFP16 GEMM単体は約21〜239倍高速化した。

## 固定した基準と環境

- 基準コミット: `01cbf633f812b9d2357f3ddfa98ffc1d746b4d0d`。
- 非同期メモリ実装: `4d81d473d`。
- WMMA・NNストリーム修正と最終再測定: `d3634a6cd48a327c5e79477b6bd58713fd3102da`。
  再測定開始時の追跡ファイル差分はゼロ。
- rocJPEG: `67ff7a307d977744efbe10e916a6d41cf26a7b2c`、Windows非AMF fork。全比較で同じDLL。
- Windows 11、Ryzen AI MAX+ 395（16コア32スレッド）、Radeon 8060S / gfx1151。
- AMDドライバ `32.0.31041.1004`、ROCm SDK `10.2.0a20260918`、HIP `7.16.26373`。
- VS 18 Insiders、Release、32並列ビルド。測定中に別の学習・GPUテスト・ビルドを実行しない。
- Tanks and Temples `truck`、251枚のJPEG、979×546。1/2縮小は489×273。
- CPU/GPUの縮小方式はLanczos2を維持。読み込み・画質設定は基準と共通。
- 基準のexe SHA-256: `40866824AA7680B907CE06E949E7E7177BA222981385E66698616F8709988E17`。
- 基準のlfs_core.dll SHA-256: `CF0DFB27BFED4FE7DD65F2322207ED6E18F7881EEAE90820675B81BE53DF4CFE`。
- rocjpeg.dll SHA-256: `781B354CF32EBAB31AEB9FEA4F0A2292ACACBF2A8B7A771772B5ED558F007503`。
- amdhip64_7.dll SHA-256: `5D7B12587955DD58EE37E2FDAB8C09D5F2ED101ECF181E88A998169138058CFC`。
- 最終exe SHA-256: `8D1DDC5AA7C3AE6575571A8D03102F63534D668729374DE544BDBE179738E16A`。
- 最終lfs_core.dll SHA-256: `F5A3CD9945CC1C8116D1BA56D52F19BB04AC7A6E158DA397A5EB4F47BEF66192`。

## truck: 確定コミットの最終比較

100,000 splats、2,000 iterationsの固定仕事量。ウォームアップ・交互実行などの条件は
次節の非同期単独比較と同じ。基準は最初に保存したruntimeを最後まで使用した。

| 条件 | 基準の中央値 | 最終版の中央値 | 時間短縮 |
|---|---:|---:|---:|
| 等倍 | 15.8405 s | 15.2460 s | 3.75% |
| 1/2、Lanczos2 | 17.0155 s | 16.2575 s | 4.45% |

| 条件 | 基準、実行1〜4 (s) | 最終版、実行1〜4 (s) |
|---|---|---|
| 等倍 | 15.762 / 15.953 / 15.919 / 15.711 | 15.179 / 15.206 / 15.286 / 15.486 |
| 1/2 | 16.971 / 17.070 / 17.060 / 16.882 | 16.320 / 16.263 / 16.252 / 16.230 |

全8組で最終版が短かった。プロセス全体の時間中央値も、等倍18.7655→18.1582 s、
1/2は19.9285→19.2048 sへ短縮した。hardware decodeは2,002〜2,004回、
CPU decodeとhot cache hitは全実行で0。先読みがあるためdecode回数はiteration数を少し上回る。

最後に表示されたlossは、等倍で基準0.0804〜0.0808 / 最終版0.0806〜0.0808、
1/2で基準0.0672〜0.0844 / 最終版0.0670〜0.0672だった。
学習中の単一画像のlossを画質同等性の根拠にせず、後述のholdout評価を用いる。
非同期単独比較とは独立した再測定であり、両表の差をWMMAのtruckへの追加効果とは解釈しない。

## truck: 非同期メモリだけを変更した比較

MRNF、100,000 splats、2,000 iterations、SH3、SH degree interval 1。
densification・評価・途中保存は無効。各条件に別プロセスの300 iterationウォームアップを実施し、
4組を基準→変更後、変更後→基準の順で交互に測定した。
GPUイベントによる診断計測を入れていない通常の学習時間を使用する。

| 条件 | 基準の中央値 | 非同期版の中央値 | 時間短縮 |
|---|---:|---:|---:|
| 等倍 | 15.6920 s | 15.3340 s | 2.28% |
| 1/2、Lanczos2 | 17.0565 s | 16.3075 s | 4.39% |

| 条件 | 基準、実行1〜4 (s) | 非同期版、実行1〜4 (s) |
|---|---|---|
| 等倍 | 15.676 / 15.708 / 15.919 / 15.620 | 15.121 / 15.301 / 15.452 / 15.367 |
| 1/2 | 16.904 / 17.137 / 17.168 / 16.976 | 16.246 / 16.632 / 16.369 / 16.168 |

8組すべてで非同期版の時間が短かった。最後に表示されたlossの範囲は、
等倍で基準0.0805〜0.0808 / 非同期版0.0806〜0.0808、
1/2で基準0.0669〜0.0674 / 非同期版0.0669〜0.0671。
学習にはGPU演算由来の非決定性があり、bitwise一致は要求しない。

## truck: 密度増加とholdout画質

同じ基準runtimeを用い、start_refine=500、refine_every=100、上限300,000 splatsで
2,000 iterationsを2組ずつ比較した。8枚おきに分離した32枚を最後に評価する。
全実行で300,000 splatsへ到達した。以下の時間は学習と最終評価を含むアプリの報告値。

| 条件 | 基準、2回 (s) | 変更後、2回 (s) | 中央値の短縮 |
|---|---|---|---:|
| 等倍 | 38.514 / 37.191 | 36.680 / 36.659 | 3.13% |
| 1/2、Lanczos2 | 29.645 / 29.877 | 28.999 / 28.810 | 2.88% |

| 条件 | 版 | PSNR (dB)、平均 | SSIM、平均 | LPIPS、平均 |
|---|---|---:|---:|---:|
| 等倍 | 基準 | 22.759245 | 0.810800 | 0.259233 |
| 等倍 | 変更後 | 22.762442 | 0.810819 | 0.259514 |
| 1/2 | 基準 | 23.807157 | 0.875414 | 0.143649 |
| 1/2 | 変更後 | 23.804324 | 0.875731 | 0.143160 |

平均PSNR差の絶対値は0.0032 dB以下、SSIMは0.000317以下、LPIPSは0.000490以下。
この短い学習で大きな画質退行は観測していない。2回ずつの結果から厳密な画質同等性や
長期学習全般の改善を主張しない。MoGe全モデルの推論時間は別途測定していない。

## メモリ・ストリーム実装

従来の `CUDART_VERSION >= 11020` はHIPのバージョン番号をCUDAの番号と比較し、
HIPで利用可能な非同期確保・解放も無効にしていた。
APIのコンパイル条件をCUDA/HIP別に判定し、実行時には選択デバイスの
`hipDeviceAttributeMemoryPoolsSupported` を確認する。
[HIPのstream-ordered allocator仕様](https://rocmdocs.amd.com/projects/HIP/en/develop/how-to/hip_runtime_api/memory_management/stream_ordered_allocator.html)
に従い、確保・利用・解放を同じストリーム、またはイベントで順序付ける。
上記ドキュメントのdevelop版と、実測したSDK/HIPバージョンは区別している。

- Tensorのbucket/exact確保と学習用scratchを `hipMallocAsync` / `hipFreeAsync` へ接続。
  確保方法を所有オブジェクトに保持し、move後も対応する解放方法を使う。
- ドライバのmemory poolは64 MiBのrelease thresholdを使用。既存のTensor bucket cache上限は維持。
- 非標準ストリームで確保した領域をlegacy/default streamが利用した場合も、
  最後のconsumerが完了してから再利用・解放する。
- 非標準ストリームへのCPU uploadは、所有するpinned stagingへコピーしてから
  Tensorのhome streamへDMAを投入する。stagingの再利用は完了イベントを待つ。
  元のCPUバッファはAPI復帰後に破棄できる。
- capture中の短命なhost stagingはgraph replayの寿命を保証できないため、明示的に未対応エラーを返す。
- HIPでも利用可能なpinned intersection counterを有効化。pool使用量の取得・trimも実際のcapabilityで判定。
- NNの出力は実行ストリーム上で確保。GEMMのbias・scale・residualにもproducer依存を設定する。
  GEMMを呼ぶ畳み込み・転置畳み込みのbiasについても同じ待ち合わせを行う。

`LFS_DISABLE_ASYNC_ALLOCATOR=1` で同期確保へ戻せる。未対応ドライバでも同期経路を使う。
所有元が不明な外部ポインタや、明示的に同期完了を必要とする読み戻し・終了処理を
無条件に非同期化していない。

## RDNA WMMA GEMM

[adelj88/rocm_wmma_gemm](https://github.com/adelj88/rocm_wmma_gemm) の
`ea3aa74fc984b9d1ef7c48b86cdcc45c93b732b2` を参照した。
wave32のfragment配置、LDSのpadding、次のK tileをレジスタへ先読みする構成を取り入れ、
LFSのepilogueへ統合した。派生ファイルにMITライセンスと著作権表記を保持する。

既存FP16 GEMMはCUDAのWMMAが無効なHIPでは、各出力を通常の積和ループで計算していた。
新経路は `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` を使う。
FP16入力・FP32累積・FP16出力を維持し、bias、各activation、scale、residual、
転置、batched/broadcast B、NCHW出力、2×2 conv-transpose scatterを扱う。
大きさに応じて128×64、64×64、32×32 tileを選び、M/N/Kの端数をマスクする。
試した128×128 tileはこのGPUの複数形状で遅くなったため採用していない。

gfx1100/1101/1102/1103/1150/1151が対象で、実機検証はgfx1151。
対象外のデバイス・小さい形状では既存経路へ戻る。
`LFS_DISABLE_HIP_WMMA=1` で比較・診断用に従来経路を選べる。
一般のFP32 GEMM、PPISPの小行列、attention内部の専用演算は精度を変えずに維持した。

同じ最終バイナリでWMMA有効/無効を4回ずつ交互に実行。
各形状5回ウォームアップ、20回のpublic `nn::gemm` 呼び出しをGPUイベントで測定した。
出力確保・bias・GELU、必要な入力転置も含む。以下は各プロセス平均の中央値。

| 用途相当 | M×N×K | 従来HIP | WMMA | 倍率 |
|---|---|---:|---:|---:|
| MoGe QKV | 1370×2304×768 | 110.0963 ms | 0.4611 ms | 238.78× |
| MoGe projection | 1370×768×768 | 3.5213 ms | 0.1619 ms | 21.75× |
| MoGe MLP up | 1370×3072×768 | 112.9328 ms | 0.6230 ms | 181.27× |
| MoGe MLP down | 1370×768×3072 | 12.0104 ms | 0.5780 ms | 20.78× |
| decoder形状、trans_a=true | 4096×256×256 | 1.6677 ms | 0.1274 ms | 13.09× |
| NN layout、trans_b=false | 1370×768×768 | 2.9455 ms | 0.1676 ms | 17.57× |
| 端数のある小行列 | 33×65×127 | 0.1336 ms | 0.0128 ms | 10.45× |

GEMM単体の改善であり、MoGe全体やtruckが同じ倍率になることは示していない。
通常のtruck rasterizerにはこのFP16 NN GEMMが主処理として含まれない。
生成したgfx1151 code objectを逆アセンブルし、`v_wmma_f32_16x16x16_f16` の18命令箇所を確認した。

## 正確性・寿命・異常系

- GPU memory: 47テスト通過。確保方法を保つmove、pending consumer後の再利用、
  legacy/non-blocking streamの依存、解放後のstream破棄、pageable upload元の寿命、
  captureの未対応エラー、未知ポインタの同期解放などを確認。
- rocJPEG/Lanczos: 42テスト通過。
- NN: 33テスト通過。CPU参照との端数・深いK・broadcast batch・融合epilogue、転置/NCHW、scatter、
  attention/convなどの既存テストと、遅延producerを使うストリーム回帰テストを実行。
  NumPy resize fixtureが存在しない1件はskip。Python weight exporterは対象から除外。
- `LFS_DISABLE_ASYNC_ALLOCATOR=1` と `LFS_DISABLE_HIP_WMMA=1` のfallback suiteも実行。
- upload順序とNN epilogueの待ち合わせ漏れは修正前に失敗を再現し、修正後に成功を確認。

CUDAのビルド・NVIDIA実機、およびgfx1151以外のAMD実機は今回検証していない。
FP32をFP16へ丸めてWMMAへ流す変更は含まれない。

## 再現

`scripts/benchmark_hip_async.ps1` は独立した2つのRelease runtimeを受け取る。
rocJPEG/HIP DLLの同一性、デコード回数、学習終了、splats数、異常ログを検査し、
引数・設定・runtime hash・各回の時間・loss・評価値を `results.json` に保存する。
一時configと生成した学習プロジェクトは終了時に削除する。

```powershell
.\scripts\benchmark_hip_async.ps1 -Dataset <truck> `
  -BaselineRuntime <baseline-Release> -CandidateRuntime <candidate-Release> `
  -BaselineCommit 01cbf633f812b9d2357f3ddfa98ffc1d746b4d0d `
  -OutputDirectory <new-results-directory> -Repetitions 4
```

`-Densify -MaxCap 300000 -Evaluate -Repetitions 2` で密度増加・holdout評価を行う。
`-DisableRocJpeg` はCPUデコーダー、`-CollectProfile` は診断計測を有効にする。
後者のタイミングは通常の学習時間と混在させない。

GPUテストは `BUILD_GPU_MEMORY_TESTS=ON`、`BUILD_GPU_NN_TESTS=ON`、
`BUILD_ROCJPEG_TESTS=ON` でLibTorchなしに構成できる。

```powershell
cmake --build build-rocm10 --config Release --parallel 32 `
  --target lichtfeld_gpu_memory_tests lichtfeld_nn_tests lichtfeld_rocjpeg_tests
ctest --test-dir build-rocm10 -C Release -R 'lichtfeld_(nn|gpu_memory|rocjpeg)' -j 1 --output-on-failure

$env:LFS_GEMM_BENCH_OUTPUT = '<new-result.json>'
.\build-rocm10\Release\lichtfeld_nn_tests.exe `
  --gtest_filter=NnGemm.DISABLED_ProductionShapesBenchmark --gtest_also_run_disabled_tests
```

WMMA無効比較は別プロセスへ `LFS_DISABLE_HIP_WMMA=1` を設定する。
入力はFP16で正確に表現できる固定乱数で、全出力のfinite確認に加え、
各production形状の64出力をdouble累積のCPU参照と比較する。
許容差はFP16出力丸めを考慮した `3e-5 + 6e-4 * abs(reference)`。

ローカルの生データは `benchmark_outputs/hip_async_wmma_final_20260922/results.json`、
`hip_async_final_20260922/results.json`、`hip_async_densify_eval_20260922/results.json`、
`hip_wmma_validation_20260922/gemm-committed-*.json` と検証XMLに保存した。
基準runtimeの一時複製、参照リポジトリのclone、逆アセンブル中間ファイル、
コンパイル途中のログは測定終了後に削除済み（合計約848 MB）。通常のRelease実行環境と測定記録は保持した。
