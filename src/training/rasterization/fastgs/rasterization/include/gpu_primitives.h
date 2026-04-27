/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstddef>

#if (defined(USE_HIP) && USE_HIP) || (defined(LFS_USE_HIP) && LFS_USE_HIP) || defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    #include <rocprim/rocprim.hpp>
#else
    #include <cub/cub.cuh>
#endif

namespace fast_lfs::rasterization::gpu_primitives {

    template <typename T>
    struct DoubleBuffer {
        T* current = nullptr;
        T* alternate = nullptr;
        int selector = 0;

        DoubleBuffer() = default;

        DoubleBuffer(T* current_, T* alternate_)
            : current(current_),
              alternate(alternate_) {
        }

        T* Current() const {
            return selector == 0 ? current : alternate;
        }

        T* Alternate() const {
            return selector == 0 ? alternate : current;
        }

        void swap() {
            selector ^= 1;
        }
    };

    template <typename T>
    inline void exclusive_sum(void* temporary_storage, size_t& storage_size, const T* input, T* output, int size) {
#if (defined(USE_HIP) && USE_HIP) || (defined(LFS_USE_HIP) && LFS_USE_HIP) || defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
        auto status = rocprim::exclusive_scan(
            temporary_storage,
            storage_size,
            input,
            output,
            T{0},
            static_cast<size_t>(size),
            rocprim::plus<T>());
#else
        auto status = cub::DeviceScan::ExclusiveSum(
            temporary_storage,
            storage_size,
            input,
            output,
            size);
#endif
        (void)status;
    }

    template <typename T>
    inline void inclusive_sum(void* temporary_storage, size_t& storage_size, const T* input, T* output, int size) {
#if (defined(USE_HIP) && USE_HIP) || (defined(LFS_USE_HIP) && LFS_USE_HIP) || defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
        auto status = rocprim::inclusive_scan(
            temporary_storage,
            storage_size,
            input,
            output,
            static_cast<size_t>(size),
            rocprim::plus<T>());
#else
        auto status = cub::DeviceScan::InclusiveSum(
            temporary_storage,
            storage_size,
            input,
            output,
            size);
#endif
        (void)status;
    }

    template <typename KeyT, typename ValueT>
    inline void sort_pairs(
        void* temporary_storage,
        size_t& storage_size,
        DoubleBuffer<KeyT>& keys,
        DoubleBuffer<ValueT>& values,
        int size,
        int begin_bit = 0,
        int end_bit = static_cast<int>(8 * sizeof(KeyT))) {
#if (defined(USE_HIP) && USE_HIP) || (defined(LFS_USE_HIP) && LFS_USE_HIP) || defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
        auto status = rocprim::radix_sort_pairs(
            temporary_storage,
            storage_size,
            keys.Current(),
            keys.Alternate(),
            values.Current(),
            values.Alternate(),
            size,
            static_cast<unsigned int>(begin_bit),
            static_cast<unsigned int>(end_bit));
#else
        auto status = cub::DeviceRadixSort::SortPairs(
            temporary_storage,
            storage_size,
            keys.Current(),
            keys.Alternate(),
            values.Current(),
            values.Alternate(),
            size,
            begin_bit,
            end_bit);
#endif
        if (temporary_storage != nullptr && size > 0 && status == cudaSuccess) {
            keys.swap();
            values.swap();
        }
        (void)status;
    }

} // namespace fast_lfs::rasterization::gpu_primitives
