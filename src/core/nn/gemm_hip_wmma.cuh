/* SPDX-FileCopyrightText: 2024 Adel Johar
 * SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: MIT
 *
 * Adapted from the RDNA fragment layout, LDS padding and register-prefetch
 * approach in https://github.com/adelj88/rocm_wmma_gemm
 * (ea3aa74fc984b9d1ef7c48b86cdcc45c93b732b2).
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#pragma once

// Included inside gemm.cu's anonymous namespace, after store_c_f16. Reuse the
// production epilogue, including activation, residual, scale and NCHW scatter.
namespace hip_wmma {

    bool supported() {
        static const bool disabled = lfs::core::environment::flag("LFS_DISABLE_HIP_WMMA");
        if (disabled) {
            return false;
        }
        int device = -1;
        if (cudaGetDevice(&device) != cudaSuccess) {
            return false;
        }
        thread_local int cached_device = -1;
        thread_local bool available = false;
        if (cached_device != device) {
            cudaDeviceProp properties{};
            if (cudaGetDeviceProperties(&properties, device) != cudaSuccess) {
                return false;
            }
            available = false;
            for (const char* arch : {"gfx1100", "gfx1101", "gfx1102", "gfx1103", "gfx1150", "gfx1151"}) {
                available |= std::strncmp(properties.gcnArchName, arch, 7) == 0;
            }
            cached_device = device;
        }
        return available;
    }

    template <int FM, int FN, int WavesM, int WavesN>
    __global__ void __launch_bounds__(WavesM* WavesN * 32)
        gemm_kernel(const __half* __restrict__ a, const __half* __restrict__ b,
                    __half* __restrict__ c, const __half* bias, int m, int n, int k,
                    long long stride_a, long long stride_b, long long stride_c,
                    bool trans_a, bool trans_b, int activation, bool trans_c,
                    const __half* residual, const __half* scale, int scatter_h, int scatter_w) {
        constexpr int BM = WavesM * FM * 16;
        constexpr int BN = WavesN * FN * 16;
        constexpr int BK = 32;
        constexpr int Threads = WavesM * WavesN * 32;
        const int tid = threadIdx.x;
        const int block_row = blockIdx.y * BM;
        const int block_col = blockIdx.x * BN;
        const int batch = blockIdx.z;
        a += batch * stride_a;
        b += batch * stride_b;
        c += batch * stride_c;
        if (residual) {
            residual += batch * stride_c;
        }

#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__)
        using HalfFragment = _Float16 __attribute__((ext_vector_type(16)));
        using FloatFragment = float __attribute__((ext_vector_type(8)));
        // Native RDNA WMMA inputs are one A row/B column per half-wave lane.
        // Both halves hold identical 16-element inputs; each lane owns 8 FP32
        // results. Padding the contraction dimension avoids LDS bank conflicts.
        __shared__ __half as[BM][BK + 8];
        __shared__ __half bs[BN][BK + 8];
        FloatFragment accum[FM][FN]{};
        __half next_a[BM * BK / Threads];
        __half next_b[BN * BK / Threads];
        const int lane = tid % 32;
        const int wave = tid / 32;
        const int wave_m = wave / WavesN;
        const int wave_n = wave % WavesN;

        auto prefetch = [&](int tile) {
#pragma unroll
            for (int i = 0; i < BM * BK / Threads; ++i) {
                const int index = tid + i * Threads;
                const int row = trans_a ? index % BM : index / BK;
                const int kk = trans_a ? index / BM : index % BK;
                const int gr = block_row + row;
                const int gk = tile + kk;
                next_a[i] = gr < m && gk < k
                                ? a[trans_a ? static_cast<long long>(gk) * m + gr
                                            : static_cast<long long>(gr) * k + gk]
                                : __float2half(0.0f);
            }
#pragma unroll
            for (int i = 0; i < BN * BK / Threads; ++i) {
                const int index = tid + i * Threads;
                const int col = trans_b ? index / BK : index % BN;
                const int kk = trans_b ? index % BK : index / BN;
                const int gc = block_col + col;
                const int gk = tile + kk;
                next_b[i] = gc < n && gk < k
                                ? b[trans_b ? static_cast<long long>(gc) * k + gk
                                            : static_cast<long long>(gk) * n + gc]
                                : __float2half(0.0f);
            }
        };
        auto commit = [&]() {
#pragma unroll
            for (int i = 0; i < BM * BK / Threads; ++i) {
                const int index = tid + i * Threads;
                as[trans_a ? index % BM : index / BK][trans_a ? index / BM : index % BK] = next_a[i];
            }
#pragma unroll
            for (int i = 0; i < BN * BK / Threads; ++i) {
                const int index = tid + i * Threads;
                bs[trans_b ? index / BK : index % BN][trans_b ? index % BK : index / BN] = next_b[i];
            }
        };
        prefetch(0);
        commit();
        __syncthreads();
        for (int tile = 0; tile < k; tile += BK) {
            const bool more = tile + BK < k;
            if (more) {
                prefetch(tile + BK);
            }
#pragma unroll
            for (int slice = 0; slice < BK; slice += 16) {
                HalfFragment af[FM];
                HalfFragment bf[FN];
#pragma unroll
                for (int i = 0; i < FM; ++i) {
#pragma unroll
                    for (int t = 0; t < 16; ++t) {
                        af[i][t] = static_cast<_Float16>(as[(wave_m * FM + i) * 16 + lane % 16][slice + t]);
                    }
                }
#pragma unroll
                for (int j = 0; j < FN; ++j) {
#pragma unroll
                    for (int t = 0; t < 16; ++t) {
                        bf[j][t] = static_cast<_Float16>(bs[(wave_n * FN + j) * 16 + lane % 16][slice + t]);
                    }
                }
#pragma unroll
                for (int i = 0; i < FM; ++i) {
#pragma unroll
                    for (int j = 0; j < FN; ++j) {
                        accum[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(af[i], bf[j], accum[i][j]);
                    }
                }
            }
            __syncthreads();
            if (more) {
                commit();
                __syncthreads();
            }
        }
#pragma unroll
        for (int i = 0; i < FM; ++i) {
#pragma unroll
            for (int j = 0; j < FN; ++j) {
#pragma unroll
                for (int t = 0; t < 8; ++t) {
                    const int row = block_row + (wave_m * FM + i) * 16 + t * 2 + lane / 16;
                    const int col = block_col + (wave_n * FN + j) * 16 + lane % 16;
                    store_c_f16(c, row, col, m, n, accum[i][j][t], bias, activation, trans_c,
                                residual, scale, scatter_h, scatter_w);
                }
            }
        }
#else
        // A fat binary may contain other architectures. Host dispatch excludes
        // them, but retain a functional implementation rather than an empty kernel.
        for (int i = tid; i < BM * BN; i += Threads) {
            const int row = block_row + i / BN;
            const int col = block_col + i % BN;
            if (row >= m || col >= n) {
                continue;
            }
            float value = 0.0f;
            for (int kk = 0; kk < k; ++kk) {
                value += __half2float(a[trans_a ? static_cast<long long>(kk) * m + row : static_cast<long long>(row) * k + kk]) *
                         __half2float(b[trans_b ? static_cast<long long>(col) * k + kk : static_cast<long long>(kk) * n + col]);
            }
            store_c_f16(c, row, col, m, n, value, bias, activation, trans_c, residual, scale, scatter_h, scatter_w);
        }
#endif
    }

    bool launch(const __half* a, const __half* b, __half* c, const __half* bias,
                int m, int n, int k, long long stride_a, long long stride_b,
                long long stride_c, int batch, bool trans_a, bool trans_b,
                int activation, bool trans_c, cudaStream_t stream,
                const __half* residual, const __half* scale, int scatter_h, int scatter_w) {
        if (m < 16 || n < 16 || k < 16 || !supported()) {
            return false;
        }
        if (m >= 96 && n >= 64) {
            gemm_kernel<2, 2, 4, 2><<<dim3((n + 63) / 64, (m + 127) / 128, batch), 256, 0, stream>>>(
                a, b, c, bias, m, n, k, stride_a, stride_b, stride_c, trans_a, trans_b,
                activation, trans_c, residual, scale, scatter_h, scatter_w);
        } else if (m >= 48 && n >= 48) {
            gemm_kernel<2, 2, 2, 2><<<dim3((n + 63) / 64, (m + 63) / 64, batch), 128, 0, stream>>>(
                a, b, c, bias, m, n, k, stride_a, stride_b, stride_c, trans_a, trans_b,
                activation, trans_c, residual, scale, scatter_h, scatter_w);
        } else {
            gemm_kernel<1, 1, 2, 2><<<dim3((n + 31) / 32, (m + 31) / 32, batch), 128, 0, stream>>>(
                a, b, c, bias, m, n, k, stride_a, stride_b, stride_c, trans_a, trans_b,
                activation, trans_c, residual, scale, scatter_h, scatter_w);
        }
        LFS_CUDA_LAUNCH_CHECK(stream, "nn.gemm.hip_wmma");
        return true;
    }

} // namespace hip_wmma
