/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda_version.hpp"
#include "config.h"
#include <cuda_runtime.h>
#include <atomic>
#include <mutex>
#include <sstream>

namespace lfs::core {

    namespace {
        constexpr int GPU_DEVICE_UNINITIALIZED = -2;

        std::atomic<int> g_selected_device{GPU_DEVICE_UNINITIALIZED};
        std::mutex g_probe_mutex;
        int g_device_count = 0;
        std::string g_probe_error;

        std::string gpu_error_string(cudaError_t err) {
            if (const char* msg = cudaGetErrorString(err); msg) {
                return msg;
            }
            return "unknown GPU runtime error";
        }
    } // namespace

    CudaVersionInfo check_cuda_version() {
        CudaVersionInfo info;

#if LFS_USE_HIP
        if (cudaRuntimeGetVersion(&info.driver_version) != cudaSuccess) {
            info.query_failed = true;
            info.supported = true;
            return info;
        }

        info.major = info.driver_version / 1000;
        info.minor = (info.driver_version % 1000) / 10;
        info.supported = true;
        return info;
#else
        if (cudaDriverGetVersion(&info.driver_version) != cudaSuccess) {
            info.query_failed = true;
            return info;
        }

        info.major = info.driver_version / 1000;
        info.minor = (info.driver_version % 1000) / 10;
        info.supported = info.driver_version >= MIN_CUDA_VERSION;

        return info;
#endif
    }

    GpuRuntimeProbe ensure_gpu_runtime_ready() {
        std::lock_guard<std::mutex> lock(g_probe_mutex);

        const int cached_device = g_selected_device.load(std::memory_order_acquire);
        if (cached_device != GPU_DEVICE_UNINITIALIZED) {
            return GpuRuntimeProbe{
                .available = cached_device >= 0,
                .device_count = g_device_count,
                .selected_device = cached_device,
                .error = g_probe_error,
            };
        }

        int device_count = 0;
        const cudaError_t count_err = cudaGetDeviceCount(&device_count);
        if (count_err != cudaSuccess) {
            g_device_count = 0;
            g_selected_device.store(-1, std::memory_order_release);
            g_probe_error = "cudaGetDeviceCount failed: " + gpu_error_string(count_err);
            cudaGetLastError(); // clear sticky runtime state
            return GpuRuntimeProbe{
                .available = false,
                .device_count = 0,
                .selected_device = -1,
                .error = g_probe_error,
            };
        }

        g_device_count = device_count;
        if (device_count <= 0) {
            g_selected_device.store(-1, std::memory_order_release);
#if LFS_USE_HIP
            g_probe_error = "No ROCm-capable GPU detected";
#else
            g_probe_error = "No CUDA-capable GPU detected";
#endif
            return GpuRuntimeProbe{
                .available = false,
                .device_count = 0,
                .selected_device = -1,
                .error = g_probe_error,
            };
        }

        int selected_device = -1;
        std::ostringstream probe_errors;

        for (int idx = 0; idx < device_count; ++idx) {
            cudaDeviceProp prop{};
            std::string device_name = std::to_string(idx);
            if (cudaGetDeviceProperties(&prop, idx) == cudaSuccess) {
                device_name = prop.name;
            }

            const cudaError_t set_err = cudaSetDevice(idx);
            if (set_err != cudaSuccess) {
                probe_errors << "[device " << idx << " (" << device_name << ")] cudaSetDevice failed: "
                             << gpu_error_string(set_err) << "; ";
                cudaGetLastError();
                continue;
            }

            size_t free_bytes = 0;
            size_t total_bytes = 0;
            const cudaError_t mem_err = cudaMemGetInfo(&free_bytes, &total_bytes);
            if (mem_err != cudaSuccess || total_bytes == 0) {
                probe_errors << "[device " << idx << " (" << device_name << ")] cudaMemGetInfo failed: "
                             << gpu_error_string(mem_err) << "; ";
                cudaGetLastError();
                continue;
            }

            selected_device = idx;
            break;
        }

        if (selected_device < 0) {
            g_selected_device.store(-1, std::memory_order_release);
            g_probe_error = probe_errors.str();
            if (g_probe_error.empty()) {
#if LFS_USE_HIP
                g_probe_error = "No usable ROCm GPU device found";
#else
                g_probe_error = "No usable CUDA GPU device found";
#endif
            }

            return GpuRuntimeProbe{
                .available = false,
                .device_count = device_count,
                .selected_device = -1,
                .error = g_probe_error,
            };
        }

        g_selected_device.store(selected_device, std::memory_order_release);
        g_probe_error.clear();
        cudaSetDevice(selected_device);
        cudaGetLastError();

        return GpuRuntimeProbe{
            .available = true,
            .device_count = device_count,
            .selected_device = selected_device,
            .error = {},
        };
    }

    bool bind_selected_gpu_device() {
        const auto probe = ensure_gpu_runtime_ready();
        if (!probe.available || probe.selected_device < 0) {
            return false;
        }

        int current_device = -1;
        const cudaError_t get_err = cudaGetDevice(&current_device);
        if (get_err == cudaSuccess && current_device == probe.selected_device) {
            return true;
        }

        const cudaError_t set_err = cudaSetDevice(probe.selected_device);
        if (set_err != cudaSuccess) {
            cudaGetLastError();
            return false;
        }

        return true;
    }

    int selected_gpu_device() {
        const int device = g_selected_device.load(std::memory_order_acquire);
        return device >= 0 ? device : -1;
    }

    std::string get_pytorch_cuda_tag(const std::string& version_hint) {
#if LFS_USE_HIP
        (void)version_hint;
        return "rocm";
#else
        // Explicit version mapping
        if (version_hint == "12.8")
            return "cu128";
        if (version_hint == "12.4")
            return "cu124";
        if (version_hint == "12.1")
            return "cu121";
        if (version_hint == "11.8")
            return "cu118";

        // Already in tag format
        if (version_hint == "cu128" || version_hint == "cu124" ||
            version_hint == "cu121" || version_hint == "cu118") {
            return version_hint;
        }

        // Auto-detect from system
        const auto info = check_cuda_version();
        if (info.query_failed) {
            return "cu128"; // Default to latest
        }

        if (info.major >= 12) {
            if (info.minor >= 8)
                return "cu128";
            if (info.minor >= 4)
                return "cu124";
            return "cu121";
        }

        if (info.major == 11 && info.minor >= 8) {
            return "cu118";
        }

        return "cu118"; // Fallback for older CUDA
#endif
    }

} // namespace lfs::core
