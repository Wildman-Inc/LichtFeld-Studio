/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <cstdio>
#include <cuda_runtime.h>

namespace {

__global__ void backend_smoke_kernel(int* out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *out = 42;
    }
}

} // namespace

int main() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        // Treat missing GPU/runtime as skipped in generic CI environments.
        std::puts("hip_backend_smoke: skipped (no visible HIP device)");
        return 0;
    }

    int* d_out = nullptr;
    int h_out = 0;
    if (cudaMalloc(&d_out, sizeof(int)) != cudaSuccess) {
        return 2;
    }

    backend_smoke_kernel<<<1, 1>>>(d_out);
    if (cudaGetLastError() != cudaSuccess) {
        cudaFree(d_out);
        return 3;
    }

    if (cudaDeviceSynchronize() != cudaSuccess) {
        cudaFree(d_out);
        return 4;
    }

    if (cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) {
        cudaFree(d_out);
        return 5;
    }
    cudaFree(d_out);

    return h_out == 42 ? 0 : 6;
}
