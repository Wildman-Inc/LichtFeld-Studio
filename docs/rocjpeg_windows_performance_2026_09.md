# Windows rocJPEG: buffer reuse and synchronization, September 2026

2026-09-22に、Windows非AMF forkとLFSの画像経路を最適化した。
Radeon 8060Sでの最終比較では、CPUデコードに対する学習時間の短縮は
等倍 **0.98%**、1/2縮小 **1.15%**。各条件4組の比較すべてでrocJPEG側が短かった。
差は小さく、このGPU・ドライバ・データセット・学習設定での結果である。
他のGPU、画像サイズ、densificationを含む学習やLinuxの性能は測定していない。

## 最終測定

- LFS実装: [`da87c5ff6`](https://github.com/Wildman-Inc/LichtFeld-Studio/commit/da87c5ff6cf9b244c98b7adc23c564efe7d08a77)。測定時の追跡ファイルの差分はゼロ。
- rocJPEG: [`67ff7a3`](https://github.com/Yasei-no-otoko/rocJPEG/commit/67ff7a307d977744efbe10e916a6d41cf26a7b2c)。LFSのFetchContent参照も同じコミット。
- Windows 11、Radeon 8060S / gfx1151、Ryzen AI MAX+ 395、AMDドライバ32.0.31041.1004。
- ROCm SDK `10.2.0a20260918`、HIP `7.16.26373`。VS 18 Insiders、Release、ビルド並列数32。
- Tanks and Temples `truck`、251 JPEG、979×546。1/2は489×273。
- MRNF、固定100,000 splats、2,000 iterations、SH3、densification・評価・途中保存なし。
- CPU/GPUともLanczos2。等倍では縮小しない。CPU側の処理は今回変更していない。
- 条件ごとに別プロセスで300 iterationのウォームアップ。その後4組をON/OFF交互の順序で実行。
- 通常の学習時間を使用し、診断タイマーやGPUイベントによる性能計測は挿入しない。
- 並行する別の学習・GPUテスト・ビルドは実行していない。

| 2,000 iterations | CPU中央値 | rocJPEG中央値 | 学習時間短縮 |
|---|---:|---:|---:|
| 等倍 | 15.8635 s | 15.7075 s | 0.98% |
| 1/2、Lanczos2 | 17.0550 s | 16.8585 s | 1.15% |

| 条件 | CPU、実行1〜4 (s) | rocJPEG、実行1〜4 (s) |
|---|---|---|
| 等倍 | 15.745 / 16.018 / 15.802 / 15.925 | 15.490 / 15.696 / 15.792 / 15.719 |
| 1/2 | 17.001 / 17.068 / 17.042 / 17.117 | 16.818 / 16.949 / 16.899 / 16.813 |

プロセス起動から終了までの中央値も、等倍ではCPU 18.8258 s / rocJPEG 18.5570 s、
1/2ではCPU 19.8927 s / rocJPEG 19.7107 sだった。等倍の第3組は学習時間差が
0.010 sと小さく、将来の各実行でも必ず速いという保証には使わない。

ONでは全実行でhardware decodeが2,002〜2,003回、CPU decodeは0回。
OFFではhardware decodeが0回、CPU decodeが2,002〜2,004回。
先読み分があるためiteration数より少し多い。全実行でhot cache hitは0、
終了時splatsは100,000。画像をデコードせず使い回す比較ではない。

再現用ハーネスは `scripts/benchmark_rocjpeg.ps1`。
ローカルの最終ログ・実行引数・バイナリSHA-256は
`benchmark_outputs/rocjpeg_optimized_final_20260922/results.json`、
集計と検証記録は `benchmark_outputs/rocjpeg_optimization_20260922/` に保存した。

## 実装した変更

1. forkの同期完了処理で、D3D12コピーのfenceを再利用するWin32イベントで先に待つ。
   完了後にHIP外部セマフォ待ちと色変換を投入する。外部メモリの同期は維持し、
   未完了のvideo/copy待ちをHIP計算キューに載せる時間を減らした。
2. D3D11完了queryをdecoder sessionで保持し、JPEGのentropy領域は`memchr`でmarkerを探す。
   JPEGテーブル・marker検証、非同期APIのsubmit/sync契約は維持する。
3. LFSでJPEG parser、HWC作業画像、Lanczos係数を保持する。係数は寸法とkernelが同じなら再計算しない。
4. 出力を既存のdecoded-frame ringに接続する。ホスト側のlease終了後も、
   consumer streamに記録したイベントが完了するまでproducerは同じバッファを書き換えない。
5. 等倍uint8はrocJPEGのRGB_PLANARからCHW出力へ直接書く。縮小uint8は
   Lanczosのfloat積算完了後に同じカーネル内で量子化し、float CHW中間画像を省く。
   Lanczos2の座標・支持幅・端処理・最後の丸め規則は維持する。
6. 新規pool割り当ての依存待ちと実際の後処理の完了待ちを残し、何も投入していないstreamの待機を省く。
   例外時には後処理をdrainしてからCPU fallbackへ戻す。

D3D11→共有テクスチャ→D3D12線形バッファのGPU内コピーは残る。
過去に試した直接共有面とCOPY queueの組み合わせは採用していない。
デコード済み画素のGPU→CPU→GPU転送はない。

バッファ再利用だけの探索測定では、等倍17.5895 s、1/2 17.9165 sで、同時に測ったCPUより遅かった。
fence待ち・query再利用・parser改善を加えた探索測定では、それぞれ15.747 s、16.781 sになった。
これらは各2回の途中版であり、単独の変更に対する寄与の測定ではない。
探索版の手動DLL配置については`provenance.json`に記録し、最終比較は固定コミットからのビルドで行った。

## ROCm 10.x HIPで確認できたこと

実機のHIP 7.16は`hipDeviceAttributeMemoryPoolsSupported=1`を返し、
default pool、`hipMallocAsync`、`hipFreeAsync`が成功した。
割り当て→書き込み→eventで別streamへ受け渡し→読み取り→eventで戻して解放、を2,000回実行し、
全回でデータ一致を確認した。アドレスの再利用は1,999回だった。

別streamにGPU処理が残る状況で10回ずつ比較したホスト呼び出し時間の中央値は、
`hipFree`が0.4473 ms、`hipFreeAsync`が0.0083 ms。
呼び出し直後に別streamがまだ実行中だった回数は、それぞれ0/10、10/10だった。
これはこのSDKとドライバでの確認であり、すべてのWindows ROCm 10.xへの保証ではない。
[HIPのstream ordered allocator仕様](https://rocmdocs.amd.com/projects/HIP/en/develop/how-to/hip_runtime_api/memory_management/stream_ordered_allocator.html)
も、stream間の依存をeventで明示する必要を説明している。

今回の製品変更は画像バッファの保持と完了依存に限定した。
LFS全体のallocatorを一括でHIP async allocationへ切り替えてはいない。
D3Dからimportした外部メモリも、通常の`hipMallocAsync`領域として扱っていない。

## Linux upstreamとの比較

`rocm-systems/develop`の確認対象は
[`58ef18c3f72b`](https://github.com/ROCm/rocm-systems/tree/58ef18c3f72beb10b10090ce3ad644372ea16eb2/projects/rocjpeg)。

- [VA-API pool](https://github.com/ROCm/rocm-systems/blob/58ef18c3f72beb10b10090ce3ad644372ea16eb2/projects/rocjpeg/src/rocjpeg_vaapi_decoder.cpp#L230)
  は形式・寸法・surface数でidle entryを再利用する。pool上限はJPEG core数から`5 * cores + 1`に設定する。
- [interop再利用](https://github.com/ROCm/rocm-systems/blob/58ef18c3f72beb10b10090ce3ad644372ea16eb2/projects/rocjpeg/src/rocjpeg_vaapi_decoder.cpp#L269)
  はVA surfaceのexport、HIP import、mappingを保持する。
  [ビルド設定](https://github.com/ROCm/rocm-systems/blob/58ef18c3f72beb10b10090ce3ad644372ea16eb2/projects/rocjpeg/CMakeLists.txt#L234)
  では再利用が既定で有効で、ROCm 7.0未満の場合に無効化する処理がある。
- [単画像decode](https://github.com/ROCm/rocm-systems/blob/58ef18c3f72beb10b10090ce3ad644372ea16eb2/projects/rocjpeg/src/rocjpeg_decoder.cpp#L237)
  は`vaSyncSurface`後にHIPでcopy/色変換を行い、最後に`hipStreamSynchronize`してsurfaceをidleに戻す。
  Linux版も同期が不要な設計ではない。
- [batch完了処理](https://github.com/ROCm/rocm-systems/blob/58ef18c3f72beb10b10090ce3ad644372ea16eb2/projects/rocjpeg/src/rocjpeg_decoder.cpp#L319)
  は各surfaceのVA完了を待ち、JPEG core数で区切るsub-batchにつきHIP同期を1回にまとめる。
- HIPはexportされたsurfaceのmappingを直接扱う。Windows版にあるD3D11/D3D12線形化の経路はない。
  今回はLinuxのソース調査のみで、Linux実機との速度比較は行っていない。

## 検証と後片付け

forkのCTestは8/8成功。13公開API、406 HIP変換ケース、異常JPEG、サイズ・形式変更、
非同期入力の寿命、逆順完了、batch失敗時の回収、保留中の破棄を含む。
LFSの専用テストは42/42成功。CPU/GPU Lanczos2比較に加え、再利用したworkspaceと従来GPU処理の
uint8完全一致、保持中の出力保護、ホストlease解放後のGPU consumer待機を確認した。

検証用HIPプログラム・ビルド・ダウンロードしたupstreamソース、ベンチマーク用設定と
学習プロジェクトは削除した。測定ログ、集計、テスト結果とこの文書は結果資料として保持する。
