# ROCm runtime license provenance

Current Windows portable packages generate `artifact-sha256-manifest.txt` from
the selected SDK's DLLs, required device archives, available SDK notices and
provenance. ZIP validation checks those exact inputs and hashes. This verifies
package integrity; it does not establish a redistribution-license audit for
that SDK. The historical notices and audited hashes below are retained as a
record of the 7.14 review and are not used to certify newer SDKs.

## Historical ROCm 7.14 audit

The vendored notices come from the exact sources recorded by the ROCm SDK's
`share/therock/therock_manifest.json`. The package preserves that manifest
under `provenance/`. For the audited 7.14.0 SDK, the pins are TheRock
`6d7f1714045261f62747f297f2f9e5f9c6d6b465`, rocm-systems
`b8c378ef2c7220a68122d13b81dab5addd61e016`, rocm-libraries
`46653de70f6d71031bed76e15525edd331318a99`, and llvm-project
`5c9bfa94a37c59923dee3c55942566db7904b659`.

- HIP runtime: `ROCm/rocm-systems` commit
  `b8c378ef2c7220a68122d13b81dab5addd61e016`,
  `projects/clr/hipamd/LICENSE.md`.
- rocRAND and hipRAND: `ROCm/rocm-libraries` commit
  `46653de70f6d71031bed76e15525edd331318a99`,
  `projects/rocrand/LICENSE.md` and `projects/hiprand/LICENSE.md`.
- ROCm kpack: `ROCm/rocm-systems` commit
  `b8c378ef2c7220a68122d13b81dab5addd61e016`, `shared/kpack/LICENSE`.
  The kpack DLL statically links msgpack-cxx 7.0.0 and zstd 1.5.7, whose
  notices are included alongside the kpack notice.
- `rocm-openblas` and `rocm-openblas64`: OpenBLAS commit
  `18638c70eff1f0d08e2833b2724deaa128d6a334`, as selected by TheRock's
  `third-party/host-blas/CMakeLists.txt`. The OpenBLAS, LAPACK, LAPACKE, and
  reference BLAS notices are all included.

`amd_comgr.dll` maps to `amd_comgr/LICENSE.txt` from the same installed ROCm
SDK root that supplies the DLL. The historical portable package configuration
required that SDK notice and provenance manifest to match the audited ROCm
7.14.0 build.
