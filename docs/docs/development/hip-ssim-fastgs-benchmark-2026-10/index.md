# Integrated HIP benchmark: AMD base vs combined SSIM + native atomics

## Result

On Radeon 8060S (`gfx1151`), the combined PR2+PR3 source was faster in the median of six fresh-process runs for both synthetic image sizes. The median below is the standard median of the six per-process medians; paired ratio is the median, round-by-round, of candidate/base process-median ratios. Values are microseconds per integrated call, with lower being faster.

| Case | AMD base, six rounds | Base median | Combined safe, six rounds | Safe median | Safe/base paired ratio | Combined native, six rounds | Native median | Native/base paired ratio |
|---|---|---:|---|---:|---:|---|---:|---:|
| 489x273 | 406.184, 409.541, 402.337, 413.815, 408.711, 413.250 | 409.126 | 396.623, 391.504, 388.930, 398.891, 397.016, 402.134 | 396.820 | 0.96903 | 392.230, 396.509, 389.485, 398.306, 394.587, 393.718 | 394.152 | 0.96554 |
| 979x546 | 2955.880, 3129.077, 3187.130, 3110.178, 3136.407, 3155.291 | 3132.742 | 3013.995, 3068.795, 3112.037, 3076.284, 3107.488, 3130.210 | 3091.886 | 0.98994 | 3057.939, 3001.724, 3063.948, 3064.167, 3017.209, 3070.090 | 3060.943 | 0.96750 |

Relative to the base median-of-six, combined native was 3.66% faster at 489x273 and 2.29% faster at 979x546. Combined safe was 3.01% faster and 1.30% faster, respectively. Paired ratios are separately shown because they answer a different question than the ratio of aggregate medians. One full-size native round was 3.45% slower than its paired base round; all six rounds are retained.

## What was measured

Each call runs public fused L1-SSIM forward and backward, then passes the resulting GPU CHW image gradient into production `blend_backward_cu`, with `grad_alpha=0`. The synthetic forward-consistent scene has 6,912 primitives / 26,784 tile instances at 489x273 and 107,136 primitives / 416,640 tile instances at 979x546. Inputs and setup are outside the timed section. Every process performs 50 warmups and records 50 batches of 64 chained calls. The primary metric is synchronized `steady_clock` wall time per call; HIP-event time on the framework current stream is retained as a secondary check. Outputs are synchronized before stopping each batch timer.

This is an integrated synthetic photometric-loss-gradient-to-rasterizer-backward segment. It is not a full training-step or dataset benchmark and does not include data loading, optimizer, densification, or the trainer loop. The scene is synthetic, not Tanks & Temples Truck. The AMD moat-port HIP CMake path is compute-only and returns before the full Studio application targets, so an exact-source headless Truck training comparison was unavailable.

## Correctness and provenance

Snapshot gates passed for both sizes across base safe, combined safe, and combined native. The image-gradient tensors were bitwise equal. All four rasterizer gradient groups were finite and nonzero; worst cross-variant relative-L2 was below `6.4e-8` (gate `1e-4`). The CPU backward-combination oracle relative-L2 was `6.47e-7` at 489x273 and `1.07e-6` at 979x546 (gate `5e-3`). The oracle checks backward combination/reduction using the context's stored partials; it does not independently validate the forward partials.

- Base source: AMD moat-port `69d080dc9995003cef7d3fea68d7789f3eb2a0ad`; executable SHA256 `905F6D303D37F24A061E6DD5EDBDAA4F67B259F67972370E783A0C6FF8D2E0EB`.
- Combined source: `cf312e294ccb6208af502bdb7df018f0607c5a02`; executable SHA256 `C04F5FDD4C10A669B98B0495587B50D9B3F735DD7D9CC87BCEFBD8B392508CEC`.
- Shared harness SHA256: `9F8E2AA73A06CEB398EE8CBBE6B3BE4A9DAA4B46ABB5A481CD588A3DC33A3637`.
- Both builds: Release `-O3 -DNDEBUG`, HIP gfx1151. Native-atomics dispatch macro was `0` on base and `1` on combined.
- Runtime API versions: HIP runtime/driver `70260201`; loaded HIP module was `amdhip64_7.dll` from System32. The harness recorded and checked the matching build's `lfs_core.dll` module path. The actual HIP module was `C:\Windows\SYSTEM32\amdhip64_7.dll`.

An earlier one-call event-timing attempt produced a negative event interval and was discarded from this comparison. None of its samples or ratios are included. The reported result comes from the rebuilt batch-64 synchronized run above, whose 36 process records each contain 50 positive finite wall and event measurements.


## Focused correctness tests

The combined build passed all three focused PR3 native-atomic tests: `ArenaOwnershipRejectsInvalidAndExternalRanges`, `NativeAndSafeAddsMatchOnArenaStorage`, and `BlendBackwardSafeAndNativeKernelsMatchWithOverlappingInstances` (3/3). The focused SSIM suite ran 25 tests: 23 passed, with one disabled performance test. Its two failures were `FusedL1SSIMTest.NoValidPadding` (loss `0`) and `MaskedFusedL1SSIMTest.ForwardBasic` (loss `-3.95753901e-08` against the test's `>0` expectation). The AMD base build's 23-test focused suite reproduced the same two failures, with 21 passes and one disabled test; there was no new failing test in the combined source.

The focused test executables were launched with generated adjacent HIP runtime copies held out and the SDK core `bin` directory omitted from `PATH`. Their process did not capture the loaded runtime module path; the explicit System32 module-path capture above belongs to the integrated benchmark harness.
## Historical dataset evidence (separate result)

An earlier full-training comparison on the same Truck dataset measured FastGS native-atomic and SSIM changes together. It compared local baseline `f5510712693f76f3c4556929290354a28f044797` with optimized local source `2dc8f6d60ae67a255682df802a47762205bc7b96`: fixed 100,000 splats, MRNF/SH3, 2,000 iterations, densification disabled, and four alternating runs. Median training time fell from 15.1845 s to 11.8850 s at 979×546 (21.73%) and from 16.2610 s to 11.3630 s at 489×273 (30.12%). These exact results are recorded in the [historical benchmark report](https://github.com/Wildman-Inc/LichtFeld-Studio/blob/5fe20bc58c1e367bcb3f8801f68d1ad092c9a8c6/docs/hip_native_atomics_ssim_performance_2026_09.md); that commit contains the report and is not the measured baseline.

A source audit of current AMD moat-port (`69d080d`) finds fused tiled/shared-memory SSIM forward and backward, plus FastGS per-tile primitive batching with shared pixel state and atomic gradient accumulation ([SSIM forward tile](https://github.com/AMD-Ecosystem/LichtFeld-Studio/blob/69d080dc9995003cef7d3fea68d7789f3eb2a0ad/src/training/kernels/ssim.cu#L456-L477), [public SSIM dispatch](https://github.com/AMD-Ecosystem/LichtFeld-Studio/blob/69d080dc9995003cef7d3fea68d7789f3eb2a0ad/src/training/kernels/ssim.cu#L1843-L1892), [FastGS tile/batches](https://github.com/AMD-Ecosystem/LichtFeld-Studio/blob/69d080dc9995003cef7d3fea68d7789f3eb2a0ad/src/training/rasterization/fastgs/rasterization/include/kernels_backward.cuh#L357-L442), [gradient atomics](https://github.com/AMD-Ecosystem/LichtFeld-Studio/blob/69d080dc9995003cef7d3fea68d7789f3eb2a0ad/src/training/rasterization/fastgs/rasterization/include/kernels_backward.cuh#L566-L577)). Similar broad structures also occur in the historical [`f551071` baseline](https://github.com/Wildman-Inc/LichtFeld-Studio/tree/f5510712693f76f3c4556929290354a28f044797). Together, the historical 21.73–30.12% improvement over that older local baseline and current combined-native aggregate medians showing 2.29–3.66% improvement over AMD moat-port (combined-safe medians show 1.30–3.01%) support the interpretation that the AMD base already had substantial optimization. This is not causal or like-for-like evidence: historical full Truck training compares older local code revisions, while the current result is a synthetic loss-to-rasterizer segment on current moat-port and combined source. The baseline, code, workload, and measured scope differ, so the performance gap cannot be attributed solely to AMD changes.

## Reproduction artifacts

The minimal harness inputs, runner, snapshot comparator, summary script, per-process logs, and full sample-vector CSVs are bundled beside this report. Reproduction still requires the HIP SDK, PyTorch ROCm package, project build graph, private dependency prefix, and Windows compiler/runtime compatibility adjustments used for this run; the bundle is not a portable one-command build. The runner takes the base and combined executable paths, output directory, Python site-packages path, private dependency prefix, and vcpkg bin path as parameters. It reproduces the timed runner and writes fresh process logs; absolute result values can vary with machine load.

Bundle files:

- [Kernel and integrated harness](/hip-ssim-fastgs-benchmark-2026-10/integrated_bench.cu)
- [Target CMake](/hip-ssim-fastgs-benchmark-2026-10/CMakeLists.txt) and [CMake injection](/hip-ssim-fastgs-benchmark-2026-10/inject.cmake)
- [Matched-run script](/hip-ssim-fastgs-benchmark-2026-10/run-integrated-matched.ps1)
- [Snapshot comparator](/hip-ssim-fastgs-benchmark-2026-10/compare-integrated-snapshots.py) and [CSV summarizer](/hip-ssim-fastgs-benchmark-2026-10/summarize-integrated.py)
- [Raw process summary](/hip-ssim-fastgs-benchmark-2026-10/raw/integrated-summary.csv) and [paired ratios](/hip-ssim-fastgs-benchmark-2026-10/raw/integrated-paired-ratios.csv)

The 36 sanitized per-process logs are in the same `raw/` folder. Focused test logs and available GTest XML reports are under `raw/tests/`.
- [Combined native-atomic test log](/hip-ssim-fastgs-benchmark-2026-10/raw/tests/native-focused.log) and [XML](/hip-ssim-fastgs-benchmark-2026-10/raw/tests/native-focused.xml)
- [Combined SSIM test log](/hip-ssim-fastgs-benchmark-2026-10/raw/tests/combined-ssim-focused.log)
- [AMD-base SSIM test log](/hip-ssim-fastgs-benchmark-2026-10/raw/tests/base-ssim-focused.log) and [XML](/hip-ssim-fastgs-benchmark-2026-10/raw/tests/base-ssim-focused.xml)
