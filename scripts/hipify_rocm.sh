#!/usr/bin/env bash
set -euo pipefail

# HIPIFY helper for LichtFeld-Studio.
# Generates HIP-translated side-by-side files under build/hipify by default.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${1:-${ROOT_DIR}/build/hipify}"
HIPIFY_BIN="${HIPIFY_BIN:-hipify-clang}"

if ! command -v "${HIPIFY_BIN}" >/dev/null 2>&1; then
  echo "error: ${HIPIFY_BIN} not found. Install HIPIFY (hipify-clang) first." >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

readarray -t TARGETS < <(
  find "${ROOT_DIR}/src/training/kernels" -maxdepth 1 -name '*.cu' -type f
  find "${ROOT_DIR}/src/training/components" -maxdepth 1 -name '*.cu' -type f
  find "${ROOT_DIR}/src/training/rasterization/gsplat" -maxdepth 1 -name '*.cu' -type f
  find "${ROOT_DIR}/src/training/rasterization/fastgs" -name '*.cu' -type f
)

if [[ ${#TARGETS[@]} -eq 0 ]]; then
  echo "No CUDA sources found for hipify."
  exit 0
fi

echo "HIPIFY output: ${OUT_DIR}"
echo "HIPIFY binary: ${HIPIFY_BIN}"

for src in "${TARGETS[@]}"; do
  rel="${src#${ROOT_DIR}/}"
  dst="${OUT_DIR}/${rel%.cu}.hip.cpp"
  mkdir -p "$(dirname "${dst}")"
  echo "hipify: ${rel} -> ${dst#${ROOT_DIR}/}"
  "${HIPIFY_BIN}" "${src}" -- -std=c++20 -I"${ROOT_DIR}/src" -I"${ROOT_DIR}/include" > "${dst}"
done

echo "HIPIFY complete: ${#TARGETS[@]} file(s)"
