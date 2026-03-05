# LichtFeld-Studio CUDA/HIP 両対応実装計画（保存＋実施手順）

## Summary
- まず計画書を `docs/plans/rocm-hip-port-plan.md` に保存。
- その後、`CUDA` 既定を維持したまま `HIP(ROCm 7.2+)` を追加。
- MI300X (`gfx942`) を含む `--offload-arch` 指定、visualizer 含むリンク完了、CLI学習スモーク、ROCmドキュメント追加までを v1 完了条件にする。

## Implementation Changes
1. **計画保存**
- `docs/plans/rocm-hip-port-plan.md` を新規作成し、本計画を保存（実装ログ追記セクション付き）。

2. **ビルド基盤切替（A/B/C）**
- ルートCMakeを `project(... LANGUAGES C CXX)` に変更。
- `LFS_GPU_BACKEND=CUDA|HIP`（既定 `CUDA`）を導入。
- CUDA時のみ `enable_language(CUDA)` / `find_package(CUDAToolkit)` / `nvidia-smi` / `TORCH_CUDA_ARCH_LIST` 実行。
- HIP時のみ `enable_language(HIP)` / `find_package(hip CONFIG REQUIRED)` 実行。
- `LFS_AMDGPU_ARCH`（既定 `gfx942`、`rocminfo` 検出優先）を追加。
- `config.h` に `LFS_USE_CUDA` / `LFS_USE_HIP` / backend識別を出力。

3. **リンク抽象化**
- `CUDA::cudart` 直リンクをやめ、抽象ターゲット（runtime/rand/driver）経由に統一。
- CUDA: `CUDA::cudart` / `CUDA::curand` / 必要時 `CUDA::cuda_driver`。
- HIP: `hip::device`（または `hip::host`）+ `hiprand`（提供ターゲットに合わせて解決）。

4. **NVIDIA専用依存無効化（F）**
- HIP時は `external/nvImageCodec` を `add_subdirectory` しない。
- IO側は feature flag で `nvcodec` 経路を無効化し、CPU decodeへフォールバック。
- CUDA向け要件（NVIDIA driver要件）は CUDA文書側へ分離。

5. **CUDA→HIP 移植（D）**
- `.cu` は維持、HIP用に `.hip/.hip.cpp` を併存。
- `hipify-clang` 実行スクリプトを追加し、対象群（training/fastgs/gsplat ほか必要最小限）を生成更新可能にする。
- `curand` 利用箇所は `hiprand` 互換へ統一。
- `gsplat` は ROCm公式フォーク差分を参照して適用。

6. **HIP VMM 実装（ユーザー指定）**
- `memory_arena` の `cuMem*` 経路を `hipMem*` 経路に移植。
- VMM可否をランタイム判定し、不可時は既存フォールバック割当へ退避。

7. **OpenGL interop 分岐（E）**
- `LFS_ENABLE_HIP_GL_INTEROP` を追加（HIP既定ON）。
- CMakeで HIP GL interop compile-check を実施、失敗時自動OFF。
- 実装で `cudaGraphics*` / `hipGraphics*` を backend分岐。
- interop無効時も既存CPUフォールバック経路で表示・学習継続。

8. **ドキュメント追加（G）**
- ROCmビルドページを追加し、以下を記載:
- 前提: Ubuntu 24.04+, ROCm 7.x, `hipcc`, `rocminfo`
- configure/build:
  `cmake -B build-hip -G Ninja -DCMAKE_BUILD_TYPE=Release -DLFS_GPU_BACKEND=HIP -DLFS_AMDGPU_ARCH=gfx942`
  `cmake --build build-hip -j`
- 実行スモーク:
  `./build-hip/LichtFeld-Studio -d <dataset> -o <output> --train --iter 5 --headless`

9. **最小スモーク（H）**
- CTestに HIP kernel 1回起動テストを追加（環境未整備時はskip）。
- 起動ログに backend / GPU名 / arch を出力。

## Test Plan
- CUDA既定ビルドが従来通り通ること。
- HIPビルドで training/fastgs/gsplat/rasterization/visualizer がリンク完了すること。
- HIP CLI学習が少なくとも数イテレーション落ちずに進むこと。
- HIP時に nvImageCodec 経路が無効化されてもビルド・起動が壊れないこと。
- HIP GL interop ON/OFF 両方で起動可能で、失敗時フォールバックが働くこと。

## Commit Plan
1. CMake backend切替導入（CUDA既定維持）
2. HIP検出・arch指定・configマクロ
3. nvImageCodec HIP無効化＋IOフォールバック
4. training/fastgs/gsplat の HIPファイル追加
5. core/rendering/io の HIP必要差分＋hiprand
6. memory_arena HIP VMM
7. HIP GL interop + visualizer統合
8. ROCm docs + HIP smoke test

## Assumptions
- 初期サポートOSは Linux（Ubuntu 24.04+）。
- ROCm 7.x 環境で `gfx942` を最小ターゲットにする。
- CUDA backendの既定挙動と既存オプション互換を最優先で保持する。

## 実装ログ
- 2026-03-05: 計画書を保存。
- 2026-03-05: CUDA/HIP バックエンド分岐の実装に着手。
- 2026-03-05: `LFS_GPU_BACKEND` / `LFS_AMDGPU_ARCH` / HIP toolchain検出のCMake分岐を実装。
- 2026-03-05: HIP時の `nvImageCodec` 自動無効化と `nvcodec_image_loader_stub.cpp` によるフォールバックを導入。
- 2026-03-05: `hip_runtime_compat.h` を拡張し、VMM(`cuMem*`互換)・GL interop・version API のHIPマッピングを追加。
- 2026-03-05: 起動ログに backend/GPU/arch を出力、CLI文言をGPU backend中立へ更新。
- 2026-03-05: ROCmビルド手順ドキュメント更新、HIP smoke test (`tests/hip_backend_smoke.cu`) を追加。
