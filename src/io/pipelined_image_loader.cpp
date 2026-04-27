/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "io/pipelined_image_loader.hpp"
#include "core/cuda/lanczos_resize/lanczos_resize.hpp"
#include "core/cuda/undistort/undistort.hpp"
#include "core/image_io.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "cuda/image_format_kernels.cuh"
#include "io/amf_image_loader.hpp"
#include "io/nvcodec_image_loader.hpp"

#include <cuda_runtime.h>
#include <stb_image.h>

#include <algorithm>
#include <fstream>
#include <random>

namespace lfs::io {

    namespace {

        constexpr int CACHE_HASH_LENGTH = 8;
        constexpr int DEFAULT_DECODER_POOL_SIZE = 8;

        struct AmfLoaderCacheEntry {
            std::shared_ptr<AmfImageLoader> instance;
            size_t owner_count = 0;
        };

        struct NvCodecLoaderCacheEntry {
            std::shared_ptr<NvCodecImageLoader> instance;
            size_t owner_count = 0;
        };

        struct HardwareDecodeLoader {
            std::shared_ptr<AmfImageLoader> amf;
            std::shared_ptr<NvCodecImageLoader> nvcodec;

            explicit operator bool() const {
                return amf || nvcodec;
            }
        };

        [[nodiscard]] size_t normalize_decoder_pool_size(size_t decoder_pool_size) {
            return decoder_pool_size > 0 ? decoder_pool_size : DEFAULT_DECODER_POOL_SIZE;
        }

        std::string generate_cache_hash() {
            static constexpr char HEX_CHARS[] = "0123456789abcdef";

            // Thread-safe: use local RNG objects to avoid data races
            thread_local std::random_device rd;
            thread_local std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(0, 15);

            std::string hash;
            hash.reserve(CACHE_HASH_LENGTH);
            for (int i = 0; i < CACHE_HASH_LENGTH; ++i) {
                hash += HEX_CHARS[dis(gen)];
            }
            return hash;
        }

        std::filesystem::path get_temp_folder() {
#ifdef _WIN32
            const char* temp = std::getenv("TEMP");
            if (!temp)
                temp = std::getenv("TMP");
            return temp ? std::filesystem::path(temp) : std::filesystem::path("C:/Temp");
#else
            return std::filesystem::path("/tmp");
#endif
        }

        [[nodiscard]] bool prefer_amf_decode_backend() {
#if defined(LFS_ENABLE_AMF) && defined(_WIN32) && \
    (defined(LFS_USE_HIP) || defined(USE_HIP) || defined(__HIP_PLATFORM_AMD__))
            return true;
#else
            return false;
#endif
        }

        std::mutex& get_amf_mutex() {
            static std::mutex mtx;
            return mtx;
        }

        std::unordered_map<size_t, AmfLoaderCacheEntry>& get_amf_loader_cache() {
            static std::unordered_map<size_t, AmfLoaderCacheEntry> instances;
            return instances;
        }

        std::mutex& get_nvcodec_mutex() {
            static std::mutex mtx;
            return mtx;
        }

        std::unordered_map<size_t, NvCodecLoaderCacheEntry>& get_nvcodec_loader_cache() {
            static std::unordered_map<size_t, NvCodecLoaderCacheEntry> instances;
            return instances;
        }

        std::shared_ptr<AmfImageLoader> acquire_amf_loader(size_t decoder_pool_size) {
            std::lock_guard<std::mutex> lock(get_amf_mutex());
            auto& instances = get_amf_loader_cache();
            const size_t requested_pool_size = normalize_decoder_pool_size(decoder_pool_size);

            if (auto it = instances.find(requested_pool_size);
                it != instances.end() && it->second.instance) {
                return it->second.instance;
            }

            auto instance = [&requested_pool_size] {
                AmfImageLoader::Options opts;
                opts.device_id = 0;
                opts.decoder_pool_size = requested_pool_size;
                return std::make_shared<AmfImageLoader>(opts);
            }();

            instances[requested_pool_size].instance = instance;
            return instance;
        }

        std::shared_ptr<NvCodecImageLoader> acquire_nvcodec_loader(size_t decoder_pool_size) {
            std::lock_guard<std::mutex> lock(get_nvcodec_mutex());
            auto& instances = get_nvcodec_loader_cache();
            const size_t requested_pool_size = normalize_decoder_pool_size(decoder_pool_size);

            if (auto it = instances.find(requested_pool_size);
                it != instances.end() && it->second.instance) {
                return it->second.instance;
            }

            auto instance = [&requested_pool_size] {
                NvCodecImageLoader::Options opts;
                opts.device_id = 0;
                opts.decoder_pool_size = requested_pool_size;
                opts.enable_fallback = true;
                return std::make_shared<NvCodecImageLoader>(opts);
            }();

            instances[requested_pool_size].instance = instance;
            return instance;
        }

        void retain_amf_loader_cache(size_t decoder_pool_size) {
            std::lock_guard<std::mutex> lock(get_amf_mutex());
            ++get_amf_loader_cache()[normalize_decoder_pool_size(decoder_pool_size)].owner_count;
        }

        void retain_nvcodec_loader_cache(size_t decoder_pool_size) {
            std::lock_guard<std::mutex> lock(get_nvcodec_mutex());
            ++get_nvcodec_loader_cache()[normalize_decoder_pool_size(decoder_pool_size)].owner_count;
        }

        void release_amf_loader_cache(size_t decoder_pool_size) {
            std::shared_ptr<AmfImageLoader> released_instance;

            {
                std::lock_guard<std::mutex> lock(get_amf_mutex());
                auto& instances = get_amf_loader_cache();
                const size_t requested_pool_size = normalize_decoder_pool_size(decoder_pool_size);
                const auto it = instances.find(requested_pool_size);
                if (it == instances.end() || it->second.owner_count == 0)
                    return;

                auto& entry = it->second;
                --entry.owner_count;
                if (entry.owner_count == 0) {
                    released_instance = std::move(entry.instance);
                    instances.erase(it);
                }
            }

            released_instance.reset();
        }

        void release_nvcodec_loader_cache(size_t decoder_pool_size) {
            std::shared_ptr<NvCodecImageLoader> released_instance;

            {
                std::lock_guard<std::mutex> lock(get_nvcodec_mutex());
                auto& instances = get_nvcodec_loader_cache();
                const size_t requested_pool_size = normalize_decoder_pool_size(decoder_pool_size);
                const auto it = instances.find(requested_pool_size);
                if (it == instances.end() || it->second.owner_count == 0)
                    return;

                auto& entry = it->second;
                --entry.owner_count;
                if (entry.owner_count == 0) {
                    released_instance = std::move(entry.instance);
                    instances.erase(it);
                }
            }

            // Drop the cache's last reference outside the mutex so teardown does not block other callers.
            released_instance.reset();
        }

        bool is_amf_available() {
            if (!prefer_amf_decode_backend())
                return false;

            static std::once_flag flag;
            static bool available = false;
            std::call_once(flag, [] { available = AmfImageLoader::is_available(); });
            return available;
        }

        bool is_nvcodec_available() {
            static std::once_flag flag;
            static bool available = false;
            std::call_once(flag, [] { available = NvCodecImageLoader::is_available(); });
            return available;
        }

        bool is_hardware_decode_available() {
            if (is_amf_available())
                return true;
            return is_nvcodec_available();
        }

        HardwareDecodeLoader acquire_hardware_decode_loader(size_t decoder_pool_size) {
            if (is_amf_available())
                return {.amf = acquire_amf_loader(decoder_pool_size)};
            if (is_nvcodec_available())
                return {.nvcodec = acquire_nvcodec_loader(decoder_pool_size)};
            return {};
        }

        void retain_hardware_decode_loader_cache(size_t decoder_pool_size) {
            if (is_amf_available())
                retain_amf_loader_cache(decoder_pool_size);
            else if (is_nvcodec_available())
                retain_nvcodec_loader_cache(decoder_pool_size);
        }

        void release_hardware_decode_loader_cache(size_t decoder_pool_size) {
            if (is_amf_available())
                release_amf_loader_cache(decoder_pool_size);
            else if (is_nvcodec_available())
                release_nvcodec_loader_cache(decoder_pool_size);
        }

        [[nodiscard]] const char* hardware_decode_backend_name() {
            if (is_amf_available())
                return "AMF";
            if (is_nvcodec_available())
                return "NVCodec";
            return "CPU";
        }

        [[nodiscard]] bool load_params_need_processing(const LoadParams& params) {
            return params.resize_factor > 1 || params.max_width > 0 || params.undistort != nullptr;
        }

        [[nodiscard]] std::string make_base_cache_key(const std::filesystem::path& path) {
            return lfs::core::path_to_utf8(path);
        }

        void apply_requested_undistort(lfs::core::Tensor& tensor, const LoadParams& params) {
            if (!params.undistort)
                return;

            const auto scaled = lfs::core::scale_undistort_params(
                *params.undistort,
                static_cast<int>(tensor.shape()[2]),
                static_cast<int>(tensor.shape()[1]));
            tensor = lfs::core::undistort_image(tensor, scaled, nullptr);
        }

        [[nodiscard]] bool is_jpeg_file_signature(const std::filesystem::path& path) {
            std::ifstream file;
            if (!lfs::core::open_file_for_read(path, std::ios::binary, file))
                return false;

            std::array<uint8_t, 3> signature{};
            if (!file.read(reinterpret_cast<char*>(signature.data()),
                           static_cast<std::streamsize>(signature.size()))) {
                return false;
            }

            return signature[0] == 0xFF && signature[1] == 0xD8 && signature[2] == 0xFF;
        }

        [[nodiscard]] lfs::core::Tensor decode_cached_rgb_tensor(
            const HardwareDecodeLoader& decoder,
            const std::shared_ptr<std::vector<uint8_t>>& jpeg_data,
            const LoadParams& params,
            const bool cached_blob_is_base) {
            const int resize_factor = cached_blob_is_base ? params.resize_factor : 1;
            const int max_width = cached_blob_is_base ? params.max_width : 0;
            auto tensor = decoder.amf
                              ? decoder.amf->load_image_from_memory_gpu(*jpeg_data, resize_factor, max_width, params.cuda_stream)
                              : decoder.nvcodec->load_image_from_memory_gpu(*jpeg_data, resize_factor, max_width, params.cuda_stream);
            if (!tensor.is_valid() || tensor.numel() == 0)
                return {};

            if (cached_blob_is_base)
                apply_requested_undistort(tensor, params);

            return tensor;
        }

        [[nodiscard]] lfs::core::Tensor load_image_hardware_decode(
            const HardwareDecodeLoader& decoder,
            const std::filesystem::path& path,
            const LoadParams& params,
            const DecodeFormat format = DecodeFormat::RGB) {
            return decoder.amf
                       ? decoder.amf->load_image_gpu(path, params.resize_factor, params.max_width, params.cuda_stream, format)
                       : decoder.nvcodec->load_image_gpu(path, params.resize_factor, params.max_width, params.cuda_stream, format);
        }

        std::tuple<uint8_t*, int, int> load_grayscale_stb(const std::filesystem::path& path) {
            int w, h, c;
            uint8_t* const data = stbi_load(lfs::core::path_to_utf8(path).c_str(), &w, &h, &c, 1);
            return {data, w, h};
        }

    } // namespace

    PipelinedImageLoader::PipelinedImageLoader(PipelinedLoaderConfig config)
        : config_(std::move(config)) {

        LOG_INFO("[PipelinedImageLoader] batch_size={}, prefetch={}, io_threads={}, cold_threads={}, resident_gpu_cache={}, resident_limit={:.1f} GB",
                 config_.jpeg_batch_size,
                 config_.prefetch_count,
                 config_.io_threads,
                 config_.cold_process_threads,
                 config_.resident_gpu_cache,
                 resident_gpu_cache_limit_bytes() / (1024.0 * 1024.0 * 1024.0));

        const bool hardware_decode_available = is_hardware_decode_available();
        if (!hardware_decode_available && config_.cold_process_threads == 0) {
            config_.cold_process_threads = 1;
            LOG_INFO("[PipelinedImageLoader] Hardware JPEG decode unavailable; enabling CPU decode worker");
        }

        if (config_.use_filesystem_cache) {
            const auto cache_base = get_temp_folder() / "LichtFeld" / "pipeline_cache";
            fs_cache_folder_ = cache_base / ("ppl_" + generate_cache_hash());

            std::error_code ec;
            std::filesystem::create_directories(fs_cache_folder_, ec);
            if (ec) {
                LOG_WARN("[PipelinedImageLoader] Cache folder creation failed: {}", ec.message());
                config_.use_filesystem_cache = false;
            }
        }

        running_ = true;

        for (size_t i = 0; i < config_.io_threads; ++i) {
            io_threads_.emplace_back([this] { prefetch_thread_func(); });
        }

        if (hardware_decode_available) {
            gpu_decode_thread_ = std::thread([this] { gpu_batch_decode_thread_func(); });
        }

        for (size_t i = 0; i < config_.cold_process_threads; ++i) {
            cold_process_threads_.emplace_back([this] { cold_process_thread_func(); });
        }

        if (hardware_decode_available) {
            retain_hardware_decode_loader_cache(config_.decoder_pool_size);
        }

        LOG_INFO("[PipelinedImageLoader] Started {} I/O, {} GPU ({}), {} cold threads",
                 config_.io_threads,
                 hardware_decode_available ? 1 : 0,
                 hardware_decode_backend_name(),
                 config_.cold_process_threads);
    }

    PipelinedImageLoader::~PipelinedImageLoader() {
        shutdown();
    }

    void PipelinedImageLoader::shutdown() {
        if (!running_.exchange(false))
            return;

        LOG_INFO("[PipelinedImageLoader] Shutting down...");

        prefetch_queue_.signal_shutdown();
        hot_queue_.signal_shutdown();
        cold_queue_.signal_shutdown();
        output_queue_.signal_shutdown();

        for (auto& t : io_threads_) {
            if (t.joinable())
                t.join();
        }
        if (gpu_decode_thread_.joinable()) {
            gpu_decode_thread_.join();
        }
        for (auto& t : cold_process_threads_) {
            if (t.joinable())
                t.join();
        }

        cudaDeviceSynchronize();
        release_hardware_decode_loader_cache(config_.decoder_pool_size);
        if (config_.use_filesystem_cache && !fs_cache_folder_.empty()) {
            std::error_code ec;
            std::filesystem::remove_all(fs_cache_folder_, ec);
        }

        LOG_INFO("[PipelinedImageLoader] Done: {} loaded, {} hits, {} misses, resident entries={}, resident={:.1f} MB, resident_hits={}, resident_misses={}, resident_evictions={}",
                 stats_.total_images_loaded,
                 stats_.hot_path_hits,
                 stats_.cold_path_misses,
                 resident_gpu_cache_.size(),
                 resident_gpu_cache_bytes_.load() / (1024.0 * 1024.0),
                 stats_.resident_gpu_cache_hits,
                 stats_.resident_gpu_cache_misses,
                 stats_.resident_gpu_cache_evictions);
    }

    void PipelinedImageLoader::prefetch(const std::vector<ImageRequest>& requests) {
        for (const auto& req : requests) {
            prefetch_queue_.push(req);
            in_flight_.fetch_add(1, std::memory_order_acq_rel);
        }
    }

    void PipelinedImageLoader::prefetch(size_t sequence_id, const std::filesystem::path& path, const LoadParams& params) {
        ImageRequest request;
        request.sequence_id = sequence_id;
        request.path = path;
        request.params = params;
        prefetch_queue_.push(std::move(request));
        in_flight_.fetch_add(1, std::memory_order_acq_rel);
    }

    ReadyImage PipelinedImageLoader::get() {
        auto result = output_queue_.pop();
        in_flight_.fetch_sub(1, std::memory_order_acq_rel);
        return result;
    }

    std::optional<ReadyImage> PipelinedImageLoader::try_get() {
        auto result = output_queue_.try_pop();
        if (result)
            in_flight_.fetch_sub(1, std::memory_order_acq_rel);
        return result;
    }

    std::optional<ReadyImage> PipelinedImageLoader::try_get_for(std::chrono::milliseconds timeout) {
        auto result = output_queue_.try_pop_for(timeout);
        if (result)
            in_flight_.fetch_sub(1, std::memory_order_acq_rel);
        return result;
    }

    size_t PipelinedImageLoader::ready_count() const {
        return output_queue_.size();
    }

    size_t PipelinedImageLoader::in_flight_count() const {
        return in_flight_.load();
    }

    void PipelinedImageLoader::clear() {
        prefetch_queue_.clear();
        hot_queue_.clear();
        cold_queue_.clear();
        output_queue_.clear();
        {
            std::lock_guard<std::mutex> lock(pending_pairs_mutex_);
            pending_pairs_.clear();
        }
        in_flight_ = 0;
    }

    PipelinedImageLoader::CacheStats PipelinedImageLoader::get_stats() const {
        CacheStats s;
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            s = stats_;
        }
        {
            std::lock_guard<std::mutex> jpeg_lock(jpeg_cache_mutex_);
            s.jpeg_cache_entries = jpeg_cache_.size();
            s.jpeg_cache_bytes = jpeg_cache_bytes_.load();
        }
        {
            std::lock_guard<std::mutex> resident_lock(resident_gpu_cache_mutex_);
            s.resident_gpu_cache_entries = resident_gpu_cache_.size();
            s.resident_gpu_cache_bytes = resident_gpu_cache_bytes_.load();
        }
        {
            std::lock_guard<std::mutex> pairs_lock(pending_pairs_mutex_);
            s.pending_pairs_count = pending_pairs_.size();
        }
        s.prefetch_queue_size = prefetch_queue_.size();
        s.hot_queue_size = hot_queue_.size();
        s.cold_queue_size = cold_queue_.size();
        s.output_queue_size = output_queue_.size();
        return s;
    }

    lfs::core::Tensor PipelinedImageLoader::load_image_immediate(
        const std::filesystem::path& path, const LoadParams& params) {
        const auto cache_key = make_cache_key(path, params);
        const auto base_key = make_base_cache_key(path);
        if (auto cached = get_from_resident_gpu_cache(cache_key)) {
            return *cached;
        }

        auto decode_cached_hit = [&](const std::shared_ptr<std::vector<uint8_t>>& jpeg_data,
                                     const bool cached_blob_is_base) -> lfs::core::Tensor {
            try {
                auto decoder = acquire_hardware_decode_loader(config_.decoder_pool_size);
                if (!decoder)
                    return {};

                auto tensor = decode_cached_rgb_tensor(decoder, jpeg_data, params, cached_blob_is_base);
                if (tensor.is_valid() && tensor.numel() > 0)
                    return tensor;
            } catch (...) {}
            return {};
        };

        if (auto jpeg_data = load_cached_jpeg_blob(cache_key)) {
            if (auto tensor = decode_cached_hit(jpeg_data, false);
                tensor.is_valid() && tensor.numel() > 0) {
                put_in_resident_gpu_cache(cache_key, tensor);
                return tensor;
            }
        }

        if (auto jpeg_data = load_cached_jpeg_blob(base_key)) {
            if (auto tensor = decode_cached_hit(jpeg_data, true);
                tensor.is_valid() && tensor.numel() > 0) {
                put_in_resident_gpu_cache(cache_key, tensor);
                return tensor;
            }
        }

        const bool needs_requested_processing = load_params_need_processing(params);
        const bool is_original_jpeg = is_jpeg_file_signature(path);
        lfs::core::Tensor decoded;

        if (is_original_jpeg) {
            auto data = std::make_shared<std::vector<uint8_t>>(read_file(path));
            put_in_jpeg_cache(base_key, data);
            save_to_fs_cache(base_key, *data);

            if (is_hardware_decode_available()) {
                try {
                    auto decoder = acquire_hardware_decode_loader(config_.decoder_pool_size);
                    auto tensor = decode_cached_rgb_tensor(decoder, data, params, true);
                    if (tensor.is_valid() && tensor.numel() > 0) {
                        put_in_resident_gpu_cache(cache_key, tensor);
                        return tensor;
                    }
                } catch (const std::exception& e) {
                    LOG_DEBUG("[PipelinedImageLoader] Immediate JPEG decode fallback for {}: {}",
                              lfs::core::path_to_utf8(path), e.what());
                }
            }
        } else {
            const std::string path_str = lfs::core::path_to_utf8(path);
            int w = 0, h = 0, ch = 0;
            unsigned char* img_data = stbi_load(path_str.c_str(), &w, &h, &ch, 3);
            const bool used_stbi = (img_data != nullptr);
            if (img_data) {
                ch = 3;
            } else {
                auto [oiio_data, ow, oh, oc] = lfs::core::load_image(path, 1, 0);
                if (!oiio_data)
                    throw std::runtime_error("Failed to decode image: " + path_str);
                img_data = oiio_data;
                w = ow;
                h = oh;
                ch = oc;
            }

            const size_t H = static_cast<size_t>(h);
            const size_t W = static_cast<size_t>(w);
            const size_t C = static_cast<size_t>(ch);

            auto cpu_tensor = lfs::core::Tensor::from_blob(
                img_data, lfs::core::TensorShape({H, W, C}),
                lfs::core::Device::CPU, lfs::core::DataType::UInt8);
            auto gpu_uint8 = cpu_tensor.to(lfs::core::Device::CUDA);
            if (used_stbi)
                stbi_image_free(img_data);
            else
                lfs::core::free_image(img_data);

            decoded = lfs::core::Tensor::zeros(
                lfs::core::TensorShape({C, H, W}),
                lfs::core::Device::CUDA, lfs::core::DataType::Float32);
            cuda::launch_uint8_hwc_to_float32_chw(
                reinterpret_cast<const uint8_t*>(gpu_uint8.data_ptr()),
                reinterpret_cast<float*>(decoded.data_ptr()),
                H, W, C, nullptr);

            if (is_nvcodec_available()) {
                try {
                    auto nvcodec = acquire_nvcodec_loader(config_.decoder_pool_size);
                    auto jpeg_bytes = nvcodec->encode_to_jpeg(decoded, config_.cache_jpeg_quality, nullptr);
                    save_to_fs_cache(base_key, jpeg_bytes);
                    auto jpeg_shared = std::make_shared<std::vector<uint8_t>>(std::move(jpeg_bytes));
                    put_in_jpeg_cache(base_key, jpeg_shared);

                    if (needs_requested_processing) {
                        auto tensor = decode_cached_rgb_tensor(
                            HardwareDecodeLoader{.nvcodec = nvcodec}, jpeg_shared, params, true);
                        if (tensor.is_valid() && tensor.numel() > 0) {
                            put_in_resident_gpu_cache(cache_key, tensor);
                            return tensor;
                        }
                    }
                } catch (const std::exception& e) {
                    LOG_DEBUG("[PipelinedImageLoader] Immediate cache write skipped for {}: {}",
                              lfs::core::path_to_utf8(path), e.what());
                }
            }
        }

        if (!decoded.is_valid() || decoded.numel() == 0 || needs_requested_processing) {
            auto [img_data, rw, rh, rc] = lfs::core::load_image(
                path, params.resize_factor, params.max_width);
            if (!img_data) {
                throw std::runtime_error("Failed to decode image: " + lfs::core::path_to_utf8(path));
            }

            const size_t H = static_cast<size_t>(rh);
            const size_t W = static_cast<size_t>(rw);
            const size_t C = static_cast<size_t>(rc);
            auto cpu_tensor = lfs::core::Tensor::from_blob(
                img_data, lfs::core::TensorShape({H, W, C}),
                lfs::core::Device::CPU, lfs::core::DataType::UInt8);
            auto gpu_uint8 = cpu_tensor.to(lfs::core::Device::CUDA);
            lfs::core::free_image(img_data);
            decoded = lfs::core::Tensor::zeros(
                lfs::core::TensorShape({C, H, W}),
                lfs::core::Device::CUDA, lfs::core::DataType::Float32);
            cuda::launch_uint8_hwc_to_float32_chw(
                reinterpret_cast<const uint8_t*>(gpu_uint8.data_ptr()),
                reinterpret_cast<float*>(decoded.data_ptr()),
                H, W, C, nullptr);
        }

        apply_requested_undistort(decoded, params);
        put_in_resident_gpu_cache(cache_key, decoded);
        return decoded;
    }

    std::string PipelinedImageLoader::make_cache_key(const std::filesystem::path& path, const LoadParams& params) const {
        auto key = lfs::core::path_to_utf8(path) + ":rf" + std::to_string(params.resize_factor) + "_mw" + std::to_string(params.max_width);
        if (params.undistort)
            key += "_ud";
        return key;
    }

    std::string PipelinedImageLoader::make_mask_cache_key(
        const std::filesystem::path& path,
        const LoadParams& params) const {
        auto key = lfs::core::path_to_utf8(path) +
                   ":mask_rf" + std::to_string(params.resize_factor) +
                   "_mw" + std::to_string(params.max_width);
        if (params.undistort)
            key += "_ud";
        return key;
    }

    std::filesystem::path PipelinedImageLoader::get_fs_cache_path(const std::string& cache_key) const {
        // Hash avoids Unicode path issues on Windows (operator/ interprets std::string as ANSI)
        return fs_cache_folder_ / (std::to_string(std::hash<std::string>{}(cache_key)) + ".jpg");
    }

    bool PipelinedImageLoader::is_jpeg_data(const std::vector<uint8_t>& data) const {
        if (data.size() < 3)
            return false;
        return data[0] == 0xFF && data[1] == 0xD8 && data[2] == 0xFF;
    }

    std::vector<uint8_t> PipelinedImageLoader::read_file(const std::filesystem::path& path) const {
        std::ifstream file;
        if (!lfs::core::open_file_for_read(path, std::ios::binary | std::ios::ate, file))
            throw std::runtime_error("Failed to open: " + lfs::core::path_to_utf8(path));

        const auto size = file.tellg();
        file.seekg(0, std::ios::beg);

        std::vector<uint8_t> buffer(size);
        if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
            throw std::runtime_error("Failed to read: " + lfs::core::path_to_utf8(path));
        }
        return buffer;
    }

    std::shared_ptr<std::vector<uint8_t>> PipelinedImageLoader::load_cached_jpeg_blob(
        const std::string& cache_key) {
        if (auto cached = get_from_jpeg_cache(cache_key))
            return cached;

        if (!config_.use_filesystem_cache)
            return {};

        const auto fs_path = get_fs_cache_path(cache_key);
        auto done_path = fs_path;
        done_path += ".done";
        if (!std::filesystem::exists(fs_path) || !std::filesystem::exists(done_path))
            return {};

        try {
            auto data = std::make_shared<std::vector<uint8_t>>(read_file(fs_path));
            put_in_jpeg_cache(cache_key, data);
            return data;
        } catch (const std::exception& e) {
            LOG_DEBUG("[PipelinedImageLoader] Cache read skipped for key {}: {}", cache_key, e.what());
            return {};
        }
    }

    std::optional<PipelinedImageLoader::CachedJpegHit> PipelinedImageLoader::find_cached_jpeg(
        const std::string& cache_key,
        const std::string& base_key) {
        if (auto cached = load_cached_jpeg_blob(cache_key)) {
            return CachedJpegHit{.data = std::move(cached), .from_base_key = false};
        }

        if (auto cached = load_cached_jpeg_blob(base_key)) {
            return CachedJpegHit{.data = std::move(cached), .from_base_key = true};
        }

        return std::nullopt;
    }

    std::shared_ptr<std::vector<uint8_t>> PipelinedImageLoader::get_from_jpeg_cache(const std::string& cache_key) {
        std::lock_guard<std::mutex> lock(jpeg_cache_mutex_);
        const auto it = jpeg_cache_.find(cache_key);
        if (it == jpeg_cache_.end())
            return nullptr;
        it->second.last_access = std::chrono::steady_clock::now();
        return it->second.data;
    }

    void PipelinedImageLoader::put_in_jpeg_cache(const std::string& cache_key, std::shared_ptr<std::vector<uint8_t>> data) {
        std::lock_guard<std::mutex> lock(jpeg_cache_mutex_);
        const size_t size = data->size();
        if (const auto it = jpeg_cache_.find(cache_key); it != jpeg_cache_.end()) {
            jpeg_cache_bytes_ -= it->second.size_bytes;
            jpeg_cache_.erase(it);
        }
        evict_jpeg_cache_if_needed(size);
        jpeg_cache_[cache_key] = JpegCacheEntry{std::move(data), std::chrono::steady_clock::now(), size};
        jpeg_cache_bytes_ += size;
    }

    void PipelinedImageLoader::put_in_jpeg_cache(const std::string& cache_key, std::vector<uint8_t>&& data) {
        put_in_jpeg_cache(cache_key, std::make_shared<std::vector<uint8_t>>(std::move(data)));
    }

    void PipelinedImageLoader::evict_jpeg_cache_if_needed(size_t required_bytes) {
        size_t target = config_.max_cache_bytes;
        const size_t available = get_available_physical_memory();
        const size_t min_free = static_cast<size_t>(get_total_physical_memory() * config_.min_free_memory_ratio);

        if (available < min_free + required_bytes) {
            target = std::min(target, jpeg_cache_bytes_.load() / 2);
        }

        while (jpeg_cache_bytes_ + required_bytes > target && !jpeg_cache_.empty()) {
            auto oldest = jpeg_cache_.begin();
            for (auto it = jpeg_cache_.begin(); it != jpeg_cache_.end(); ++it) {
                if (it->second.last_access < oldest->second.last_access) {
                    oldest = it;
                }
            }
            jpeg_cache_bytes_ -= oldest->second.size_bytes;
            jpeg_cache_.erase(oldest);
        }
    }

    size_t PipelinedImageLoader::resident_gpu_cache_limit_bytes() const {
        if (!config_.resident_gpu_cache) {
            return 0;
        }
        return config_.resident_gpu_cache_max_bytes > 0
                   ? config_.resident_gpu_cache_max_bytes
                   : config_.max_cache_bytes;
    }

    std::optional<lfs::core::Tensor> PipelinedImageLoader::get_from_resident_gpu_cache(
        const std::string& cache_key) {
        if (!config_.resident_gpu_cache) {
            return std::nullopt;
        }

        std::optional<lfs::core::Tensor> tensor;
        {
            std::lock_guard<std::mutex> lock(resident_gpu_cache_mutex_);
            const auto it = resident_gpu_cache_.find(cache_key);
            if (it != resident_gpu_cache_.end() &&
                it->second.tensor.is_valid() &&
                it->second.tensor.device() == lfs::core::Device::CUDA) {
                it->second.last_access = std::chrono::steady_clock::now();
                tensor = it->second.tensor;
            }
        }

        std::lock_guard<std::mutex> stats_lock(stats_mutex_);
        if (tensor) {
            ++stats_.resident_gpu_cache_hits;
        } else {
            ++stats_.resident_gpu_cache_misses;
        }
        return tensor;
    }

    void PipelinedImageLoader::put_in_resident_gpu_cache(
        const std::string& cache_key,
        const lfs::core::Tensor& tensor) {
        if (!config_.resident_gpu_cache ||
            !tensor.is_valid() ||
            tensor.numel() == 0 ||
            tensor.device() != lfs::core::Device::CUDA) {
            return;
        }

        const size_t tensor_bytes = tensor.bytes();
        const size_t limit_bytes = resident_gpu_cache_limit_bytes();
        if (tensor_bytes == 0 || (limit_bytes > 0 && tensor_bytes > limit_bytes)) {
            return;
        }

        size_t evictions = 0;
        {
            std::lock_guard<std::mutex> lock(resident_gpu_cache_mutex_);
            size_t current_bytes = resident_gpu_cache_bytes_.load(std::memory_order_relaxed);

            if (const auto it = resident_gpu_cache_.find(cache_key); it != resident_gpu_cache_.end()) {
                current_bytes -= std::min(current_bytes, it->second.size_bytes);
                resident_gpu_cache_.erase(it);
            }

            while (limit_bytes > 0 &&
                   current_bytes + tensor_bytes > limit_bytes &&
                   !resident_gpu_cache_.empty()) {
                auto oldest = resident_gpu_cache_.begin();
                for (auto it = resident_gpu_cache_.begin(); it != resident_gpu_cache_.end(); ++it) {
                    if (it->second.last_access < oldest->second.last_access) {
                        oldest = it;
                    }
                }
                current_bytes -= std::min(current_bytes, oldest->second.size_bytes);
                resident_gpu_cache_.erase(oldest);
                ++evictions;
            }

            resident_gpu_cache_[cache_key] = ResidentTensorEntry{
                tensor,
                std::chrono::steady_clock::now(),
                tensor_bytes};
            current_bytes += tensor_bytes;
            resident_gpu_cache_bytes_.store(current_bytes, std::memory_order_relaxed);
        }

        if (evictions > 0) {
            std::lock_guard<std::mutex> stats_lock(stats_mutex_);
            stats_.resident_gpu_cache_evictions += evictions;
        }
    }

    void PipelinedImageLoader::save_to_fs_cache(const std::string& cache_key, const std::vector<uint8_t>& data) {
        if (!config_.use_filesystem_cache)
            return;

        std::lock_guard<std::mutex> lock(fs_cache_mutex_);
        if (files_being_written_.contains(cache_key))
            return;
        files_being_written_.insert(cache_key);

        const auto path = get_fs_cache_path(cache_key);
        std::ofstream file;
        if (lfs::core::open_file_for_write(path, std::ios::binary, file)) {
            file.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
            if (!file.good()) {
                LOG_WARN("[PipelinedImageLoader] Failed to write cache: {}", lfs::core::path_to_utf8(path));
            } else {
                file.close();
                auto done_path = path;
                done_path += ".done";
                std::ofstream done_file;
                if (!lfs::core::open_file_for_write(done_path, done_file) || !done_file.good()) {
                    LOG_WARN("[PipelinedImageLoader] Failed to create .done marker: {}", lfs::core::path_to_utf8(path));
                }
            }
        } else {
            LOG_WARN("[PipelinedImageLoader] Failed to open cache file for writing: {}", lfs::core::path_to_utf8(path));
        }
        files_being_written_.erase(cache_key);
    }

    void PipelinedImageLoader::try_complete_pair(
        size_t sequence_id,
        std::optional<lfs::core::Tensor> image,
        std::optional<lfs::core::Tensor> mask,
        cudaStream_t stream) {

        std::lock_guard<std::mutex> lock(pending_pairs_mutex_);
        auto& pair = pending_pairs_[sequence_id];

        if (image)
            pair.image = std::move(*image);
        if (mask)
            pair.mask = std::move(*mask);
        if (stream)
            pair.stream = stream;

        const bool image_ready = pair.image.has_value();
        const bool mask_has_value = pair.mask.has_value();
        const bool mask_ready = !pair.mask_expected || mask_has_value;

        if (image_ready && mask_ready) {
            output_queue_.push({sequence_id,
                                std::move(*pair.image),
                                mask_has_value ? std::optional(std::move(*pair.mask)) : std::nullopt,
                                pair.stream});
            pending_pairs_.erase(sequence_id);

            std::lock_guard<std::mutex> stats_lock(stats_mutex_);
            ++stats_.total_images_loaded;
            if (mask_has_value) {
                ++stats_.masks_loaded;
            }
        }
    }

    void PipelinedImageLoader::prefetch_thread_func() {
        const bool hardware_decode_available = is_hardware_decode_available();
        const auto enqueue_decode = [this, hardware_decode_available](PrefetchedImage item) {
            if (hardware_decode_available) {
                hot_queue_.push(std::move(item));
            } else {
                item.jpeg_data.reset();
                item.needs_processing = true;
                cold_queue_.push(std::move(item));
            }
        };

        while (running_) {
            ImageRequest request;
            try {
                request = prefetch_queue_.pop();
            } catch (const std::runtime_error&) {
                break;
            }

            {
                std::lock_guard<std::mutex> lock(pending_pairs_mutex_);
                pending_pairs_[request.sequence_id].mask_expected =
                    request.mask_path.has_value() || request.extract_alpha_as_mask;
            }

            if (request.extract_alpha_as_mask) {
                const auto rgb_key = make_cache_key(request.path, request.params);
                const auto alpha_key = make_mask_cache_key(request.path, request.params);
                auto resident_rgb = get_from_resident_gpu_cache(rgb_key);
                auto resident_alpha = get_from_resident_gpu_cache(alpha_key);
                if (resident_rgb && resident_alpha) {
                    try_complete_pair(
                        request.sequence_id,
                        std::move(resident_rgb),
                        std::move(resident_alpha),
                        nullptr);
                    continue;
                }

                auto cached_rgb = get_from_jpeg_cache(rgb_key);
                auto cached_alpha = get_from_jpeg_cache(alpha_key);

                if (cached_rgb && cached_alpha) {
                    PrefetchedImage img_item;
                    img_item.sequence_id = request.sequence_id;
                    img_item.path = request.path;
                    img_item.cache_key = rgb_key;
                    img_item.jpeg_data = cached_rgb;
                    img_item.is_cache_hit = true;
                    enqueue_decode(std::move(img_item));

                    PrefetchedImage mask_item;
                    mask_item.sequence_id = request.sequence_id;
                    mask_item.path = request.path;
                    mask_item.cache_key = alpha_key;
                    mask_item.jpeg_data = cached_alpha;
                    mask_item.is_mask = true;
                    mask_item.is_cache_hit = true;
                    enqueue_decode(std::move(mask_item));

                    std::lock_guard<std::mutex> lock(stats_mutex_);
                    ++stats_.hot_path_hits;
                } else {
                    PrefetchedImage result;
                    result.sequence_id = request.sequence_id;
                    result.path = request.path;
                    result.params = request.params;
                    result.cache_key = rgb_key;
                    result.alpha_as_mask = true;
                    result.alpha_mask_params = request.alpha_mask_params;
                    result.needs_processing = true;
                    result.undistort = request.undistort;

                    cold_queue_.push(std::move(result));
                    std::lock_guard<std::mutex> lock(stats_mutex_);
                    ++stats_.cold_path_misses;
                }
                continue;
            }

            bool image_delivered_from_resident_cache = false;
            PrefetchedImage result;
            result.sequence_id = request.sequence_id;
            result.path = request.path;
            result.params = request.params;
            result.cache_key = make_cache_key(request.path, request.params);
            result.is_mask = false;
            result.undistort = request.undistort;

            if (auto resident_image = get_from_resident_gpu_cache(result.cache_key)) {
                try_complete_pair(
                    request.sequence_id,
                    std::move(resident_image),
                    std::nullopt,
                    nullptr);
                image_delivered_from_resident_cache = true;
            }

            if (!image_delivered_from_resident_cache) {
                try {
                    const bool needs_requested_processing = load_params_need_processing(request.params);
                    const auto base_key = make_base_cache_key(request.path);

                    if (auto cached = find_cached_jpeg(result.cache_key, base_key)) {
                        result.jpeg_data = std::move(cached->data);
                        result.is_cache_hit = true;
                        result.needs_processing = cached->from_base_key && needs_requested_processing;
                        enqueue_decode(std::move(result));
                        std::lock_guard<std::mutex> lock(stats_mutex_);
                        ++stats_.hot_path_hits;
                    } else {
                        result.raw_bytes = read_file(request.path);
                        result.is_original_jpeg = is_jpeg_data(result.raw_bytes);
                        result.is_cache_hit = false;

                        {
                            std::lock_guard<std::mutex> lock(stats_mutex_);
                            stats_.total_bytes_read += result.raw_bytes.size();
                        }

                        if (result.is_original_jpeg && !needs_requested_processing) {
                            auto data = std::make_shared<std::vector<uint8_t>>(std::move(result.raw_bytes));
                            put_in_jpeg_cache(result.cache_key, data);
                            result.jpeg_data = data;
                            result.is_cache_hit = true;
                            enqueue_decode(std::move(result));
                            std::lock_guard<std::mutex> lock(stats_mutex_);
                            ++stats_.hot_path_hits;
                        } else {
                            result.needs_processing = true;
                            cold_queue_.push(std::move(result));
                            std::lock_guard<std::mutex> lock(stats_mutex_);
                            ++stats_.cold_path_misses;
                        }
                    }
                } catch (const std::exception& e) {
                    LOG_ERROR("[PipelinedImageLoader] Prefetch error {}: {}", lfs::core::path_to_utf8(request.path), e.what());
                    {
                        std::lock_guard<std::mutex> lock(pending_pairs_mutex_);
                        pending_pairs_.erase(request.sequence_id);
                    }
                    in_flight_.fetch_sub(1, std::memory_order_acq_rel);
                    continue;
                }
            } else if (!request.mask_path) {
                continue;
            }

            if (request.mask_path) {
                PrefetchedImage mask_result;
                mask_result.sequence_id = request.sequence_id;
                mask_result.path = *request.mask_path;
                mask_result.params = request.params;
                mask_result.cache_key = make_mask_cache_key(*request.mask_path, request.params);
                mask_result.is_mask = true;
                mask_result.mask_params = request.mask_params;
                mask_result.undistort = request.undistort;

                try {
                    if (auto resident_mask = get_from_resident_gpu_cache(mask_result.cache_key)) {
                        try_complete_pair(
                            request.sequence_id,
                            std::nullopt,
                            std::move(resident_mask),
                            nullptr);
                        std::lock_guard<std::mutex> lock(stats_mutex_);
                        ++stats_.mask_cache_hits;
                    } else if (auto cached = get_from_jpeg_cache(mask_result.cache_key)) {
                        mask_result.jpeg_data = cached;
                        mask_result.is_cache_hit = true;
                        enqueue_decode(std::move(mask_result));
                        std::lock_guard<std::mutex> lock(stats_mutex_);
                        ++stats_.mask_cache_hits;
                    } else if (config_.use_filesystem_cache) {
                        const auto fs_path = get_fs_cache_path(mask_result.cache_key);
                        auto done_path = fs_path;
                        done_path += ".done";
                        if (std::filesystem::exists(fs_path) && std::filesystem::exists(done_path)) {
                            auto data = std::make_shared<std::vector<uint8_t>>(read_file(fs_path));
                            put_in_jpeg_cache(mask_result.cache_key, data);
                            mask_result.jpeg_data = data;
                            mask_result.is_cache_hit = true;
                            enqueue_decode(std::move(mask_result));
                            std::lock_guard<std::mutex> lock(stats_mutex_);
                            ++stats_.mask_cache_hits;
                        } else {
                            mask_result.raw_bytes = read_file(*request.mask_path);
                            mask_result.is_original_jpeg = is_jpeg_data(mask_result.raw_bytes);
                            mask_result.is_cache_hit = false;

                            {
                                std::lock_guard<std::mutex> lock(stats_mutex_);
                                stats_.total_bytes_read += mask_result.raw_bytes.size();
                            }

                            mask_result.needs_processing = true;
                            cold_queue_.push(std::move(mask_result));
                            std::lock_guard<std::mutex> lock(stats_mutex_);
                            ++stats_.mask_cache_misses;
                        }
                    } else {
                        mask_result.raw_bytes = read_file(*request.mask_path);
                        mask_result.is_original_jpeg = is_jpeg_data(mask_result.raw_bytes);
                        mask_result.is_cache_hit = false;

                        {
                            std::lock_guard<std::mutex> lock(stats_mutex_);
                            stats_.total_bytes_read += mask_result.raw_bytes.size();
                        }

                        mask_result.needs_processing = true;
                        cold_queue_.push(std::move(mask_result));
                        std::lock_guard<std::mutex> lock(stats_mutex_);
                        ++stats_.mask_cache_misses;
                    }
                } catch (const std::exception& e) {
                    LOG_WARN("[PipelinedImageLoader] Mask prefetch error {}: {} - continuing without mask",
                             lfs::core::path_to_utf8(*request.mask_path), e.what());
                    std::lock_guard<std::mutex> lock(pending_pairs_mutex_);
                    if (auto it = pending_pairs_.find(request.sequence_id); it != pending_pairs_.end()) {
                        it->second.mask_expected = false;
                        if (it->second.image.has_value()) {
                            output_queue_.push({request.sequence_id,
                                                std::move(*it->second.image),
                                                std::nullopt,
                                                it->second.stream});
                            pending_pairs_.erase(it);
                        }
                    }
                }
            }
        }
    }

    void PipelinedImageLoader::gpu_batch_decode_thread_func() {
        std::vector<PrefetchedImage> batch;
        batch.reserve(config_.jpeg_batch_size);

        while (running_) {
            batch.clear();
            const auto deadline = std::chrono::steady_clock::now() + config_.batch_collect_timeout;

            try {
                auto first = hot_queue_.try_pop_for(config_.output_wait_timeout);
                if (!first)
                    continue;
                batch.push_back(std::move(*first));
            } catch (const std::runtime_error&) {
                break;
            }

            while (batch.size() < config_.jpeg_batch_size) {
                const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
                    deadline - std::chrono::steady_clock::now());
                if (remaining.count() <= 0)
                    break;
                auto item = hot_queue_.try_pop_for(remaining);
                if (!item)
                    break;
                batch.push_back(std::move(*item));
            }

            if (batch.empty())
                continue;

            try {
                auto decoder = acquire_hardware_decode_loader(config_.decoder_pool_size);
                if (!decoder)
                    throw std::runtime_error("No hardware JPEG decoder available");

                for (size_t i = 0; i < batch.size(); ++i) {
                    try {
                        if (batch[i].is_mask) {
                            auto mask_tensor = decoder.amf
                                                   ? decoder.amf->load_image_from_memory_gpu(
                                                         *batch[i].jpeg_data, 1, 0, nullptr, DecodeFormat::Grayscale)
                                                   : decoder.nvcodec->load_image_from_memory_gpu(
                                                         *batch[i].jpeg_data, 1, 0, nullptr, DecodeFormat::Grayscale);

                            if (!mask_tensor.is_valid() || mask_tensor.numel() == 0) {
                                LOG_WARN("[PipelinedImageLoader] GPU mask decode failed for {}",
                                         lfs::core::path_to_utf8(batch[i].path));
                                throw std::runtime_error("Invalid mask tensor");
                            }

                            put_in_resident_gpu_cache(batch[i].cache_key, mask_tensor);
                            try_complete_pair(batch[i].sequence_id, std::nullopt, std::move(mask_tensor), nullptr);

                        } else {
                            const bool decode_from_base_cache = batch[i].needs_processing;
                            auto tensor = decode_cached_rgb_tensor(
                                decoder, batch[i].jpeg_data, batch[i].params, decode_from_base_cache);

                            if (!tensor.is_valid() || tensor.numel() == 0) {
                                LOG_WARN("[PipelinedImageLoader] GPU decode failed for {}",
                                         lfs::core::path_to_utf8(batch[i].path));
                                throw std::runtime_error("Invalid tensor");
                            }

                            if (decode_from_base_cache && decoder.nvcodec) {
                                try {
                                    auto jpeg_bytes = decoder.nvcodec->encode_to_jpeg(
                                        tensor, config_.cache_jpeg_quality, batch[i].params.cuda_stream);
                                    save_to_fs_cache(batch[i].cache_key, jpeg_bytes);
                                    put_in_jpeg_cache(
                                        batch[i].cache_key,
                                        std::make_shared<std::vector<uint8_t>>(std::move(jpeg_bytes)));
                                } catch (const std::exception& e) {
                                    LOG_DEBUG("[PipelinedImageLoader] Derived cache write skipped for {}: {}",
                                              lfs::core::path_to_utf8(batch[i].path), e.what());
                                }
                            }

                            put_in_resident_gpu_cache(batch[i].cache_key, tensor);
                            try_complete_pair(batch[i].sequence_id, std::move(tensor), std::nullopt, nullptr);
                        }
                    } catch (const std::exception&) {
                        auto& item = batch[i];
                        item.is_cache_hit = false;
                        item.needs_processing = true;
                        if (item.raw_bytes.empty()) {
                            try {
                                item.raw_bytes = read_file(item.path);
                            } catch (...) {
                                // Clean up pending_pairs_ to prevent memory leak
                                std::lock_guard<std::mutex> lock(pending_pairs_mutex_);
                                if (item.is_mask) {
                                    if (auto it = pending_pairs_.find(item.sequence_id); it != pending_pairs_.end()) {
                                        it->second.mask_expected = false;
                                        if (it->second.image.has_value()) {
                                            output_queue_.push({item.sequence_id, std::move(*it->second.image),
                                                                std::nullopt, it->second.stream});
                                            pending_pairs_.erase(it);
                                        }
                                    }
                                } else {
                                    pending_pairs_.erase(item.sequence_id);
                                    in_flight_.fetch_sub(1, std::memory_order_acq_rel);
                                }
                                continue;
                            }
                        }
                        cold_queue_.push(std::move(item));
                    }
                }

                std::lock_guard<std::mutex> lock(stats_mutex_);
                ++stats_.gpu_batch_decodes;
                stats_.total_decode_calls += batch.size();

            } catch (const std::exception& e) {
                LOG_ERROR("[PipelinedImageLoader] Batch decode error: {}", e.what());
                for (auto& item : batch) {
                    item.is_cache_hit = false;
                    item.needs_processing = true;
                    if (item.raw_bytes.empty()) {
                        try {
                            item.raw_bytes = read_file(item.path);
                        } catch (...) {
                            // Clean up pending_pairs_ to prevent memory leak
                            std::lock_guard<std::mutex> lock(pending_pairs_mutex_);
                            if (item.is_mask) {
                                if (auto it = pending_pairs_.find(item.sequence_id); it != pending_pairs_.end()) {
                                    it->second.mask_expected = false;
                                    if (it->second.image.has_value()) {
                                        output_queue_.push({item.sequence_id, std::move(*it->second.image),
                                                            std::nullopt, it->second.stream});
                                        pending_pairs_.erase(it);
                                    }
                                }
                            } else {
                                pending_pairs_.erase(item.sequence_id);
                                in_flight_.fetch_sub(1, std::memory_order_acq_rel);
                            }
                            continue;
                        }
                    }
                    cold_queue_.push(std::move(item));
                }
            }
        }
    }

    void PipelinedImageLoader::cold_process_thread_func() {
        const auto decoder = acquire_hardware_decode_loader(config_.decoder_pool_size);
        const auto nvcodec = decoder.nvcodec;

        while (running_) {
            PrefetchedImage item;
            try {
                item = cold_queue_.pop();
            } catch (const std::runtime_error&) {
                break;
            }

            try {
                if (item.alpha_as_mask) {
                    auto [img_data, width, height, channels] = lfs::core::load_image_with_alpha(
                        item.path, item.params.resize_factor, item.params.max_width);

                    if (!img_data || channels != 4)
                        throw std::runtime_error("Failed to load RGBA image");

                    const size_t H = static_cast<size_t>(height);
                    const size_t W = static_cast<size_t>(width);

                    auto cpu_tensor = lfs::core::Tensor::from_blob(
                        img_data, lfs::core::TensorShape({H, W, 4}),
                        lfs::core::Device::CPU, lfs::core::DataType::UInt8);
                    auto gpu_uint8 = cpu_tensor.to(lfs::core::Device::CUDA);
                    lfs::core::free_image(img_data);

                    auto rgb = lfs::core::Tensor::zeros(
                        lfs::core::TensorShape({3, H, W}),
                        lfs::core::Device::CUDA, lfs::core::DataType::Float32);
                    auto alpha = lfs::core::Tensor::zeros(
                        lfs::core::TensorShape({H, W}),
                        lfs::core::Device::CUDA, lfs::core::DataType::Float32);

                    cuda::launch_uint8_rgba_split_to_float32_rgb_and_alpha(
                        gpu_uint8.ptr<uint8_t>(), rgb.ptr<float>(), alpha.ptr<float>(),
                        H, W, nullptr);

                    gpu_uint8 = lfs::core::Tensor();

                    float* const alpha_ptr = alpha.ptr<float>();
                    if (item.alpha_mask_params.invert) {
                        cuda::launch_mask_invert(alpha_ptr, H, W, nullptr);
                    }
                    if (item.alpha_mask_params.threshold > 0) {
                        cuda::launch_mask_threshold(alpha_ptr, H, W, item.alpha_mask_params.threshold, nullptr);
                    }

                    if (item.undistort) {
                        const auto scaled = lfs::core::scale_undistort_params(
                            *item.undistort, static_cast<int>(W), static_cast<int>(H));
                        rgb = lfs::core::undistort_image(rgb, scaled, nullptr);
                        alpha = lfs::core::undistort_mask(alpha, scaled, nullptr);
                    }

                    const auto alpha_key = make_mask_cache_key(item.path, item.params);
                    if (nvcodec) {
                        try {
                            auto rgb_jpeg = nvcodec->encode_to_jpeg(rgb, config_.cache_jpeg_quality, nullptr);
                            save_to_fs_cache(item.cache_key, rgb_jpeg);
                            put_in_jpeg_cache(item.cache_key,
                                              std::make_shared<std::vector<uint8_t>>(std::move(rgb_jpeg)));

                            auto alpha_jpeg = nvcodec->encode_grayscale_to_jpeg(
                                alpha, config_.cache_jpeg_quality, nullptr);
                            save_to_fs_cache(alpha_key, alpha_jpeg);
                            put_in_jpeg_cache(alpha_key,
                                              std::make_shared<std::vector<uint8_t>>(std::move(alpha_jpeg)));
                        } catch (const std::exception&) {
                        }
                    }

                    if (const cudaError_t err = cudaStreamSynchronize(nullptr); err != cudaSuccess) {
                        throw std::runtime_error(std::string("CUDA sync failed: ") + cudaGetErrorString(err));
                    }

                    put_in_resident_gpu_cache(item.cache_key, rgb);
                    put_in_resident_gpu_cache(alpha_key, alpha);
                    try_complete_pair(item.sequence_id, std::move(rgb), std::move(alpha), nullptr);

                } else if (item.is_mask) {
                    lfs::core::Tensor mask_tensor;
                    bool used_gpu = false;

                    if (decoder) {
                        try {
                            mask_tensor = load_image_hardware_decode(
                                decoder, item.path, item.params, DecodeFormat::Grayscale);
                            used_gpu = true;
                        } catch (const std::exception&) {
                        }
                    }

                    if (!used_gpu) {
                        const auto [gray_data, src_w, src_h] = load_grayscale_stb(item.path);
                        if (!gray_data)
                            throw std::runtime_error("Failed to decode mask");

                        int target_w = src_w;
                        int target_h = src_h;
                        if (item.params.resize_factor > 1) {
                            target_w /= item.params.resize_factor;
                            target_h /= item.params.resize_factor;
                        }
                        const int max_w = item.params.max_width;
                        if (max_w > 0 && (target_w > max_w || target_h > max_w)) {
                            if (target_w > target_h) {
                                target_h = std::max(1, max_w * target_h / target_w);
                                target_w = max_w;
                            } else {
                                target_w = std::max(1, max_w * target_w / target_h);
                                target_h = max_w;
                            }
                        }

                        const auto cpu_tensor = lfs::core::Tensor::from_blob(
                            gray_data, lfs::core::TensorShape({static_cast<size_t>(src_h), static_cast<size_t>(src_w)}),
                            lfs::core::Device::CPU, lfs::core::DataType::UInt8);
                        const auto gpu_uint8 = cpu_tensor.to(lfs::core::Device::CUDA);
                        stbi_image_free(gray_data);

                        if (target_w != src_w || target_h != src_h) {
                            mask_tensor = lfs::core::lanczos_resize_grayscale(gpu_uint8, target_h, target_w, 2, nullptr);
                        } else {
                            mask_tensor = lfs::core::Tensor::zeros(
                                lfs::core::TensorShape({static_cast<size_t>(target_h), static_cast<size_t>(target_w)}),
                                lfs::core::Device::CUDA, lfs::core::DataType::Float32);
                            cuda::launch_uint8_hw_to_float32_hw(
                                gpu_uint8.ptr<uint8_t>(), mask_tensor.ptr<float>(), target_h, target_w, nullptr);
                        }
                        cudaStreamSynchronize(nullptr);
                    }

                    const size_t H = mask_tensor.shape()[0];
                    const size_t W = mask_tensor.shape()[1];
                    float* const mask_ptr = static_cast<float*>(mask_tensor.data_ptr());

                    if (item.mask_params.invert) {
                        cuda::launch_mask_invert(mask_ptr, H, W, nullptr);
                    }
                    if (item.mask_params.threshold > 0) {
                        cuda::launch_mask_threshold(mask_ptr, H, W, item.mask_params.threshold, nullptr);
                    }
                    if (item.undistort) {
                        const auto scaled = lfs::core::scale_undistort_params(
                            *item.undistort,
                            static_cast<int>(W), static_cast<int>(H));
                        mask_tensor = lfs::core::undistort_mask(mask_tensor, scaled, nullptr);
                    }

                    if (nvcodec) {
                        try {
                            auto jpeg_bytes = nvcodec->encode_grayscale_to_jpeg(
                                mask_tensor, config_.cache_jpeg_quality, nullptr);
                            save_to_fs_cache(item.cache_key, jpeg_bytes);
                            put_in_jpeg_cache(item.cache_key,
                                              std::make_shared<std::vector<uint8_t>>(std::move(jpeg_bytes)));
                        } catch (const std::exception&) {
                        }
                    }

                    if (const cudaError_t err = cudaStreamSynchronize(nullptr); err != cudaSuccess) {
                        throw std::runtime_error(std::string("CUDA sync failed: ") + cudaGetErrorString(err));
                    }

                    put_in_resident_gpu_cache(item.cache_key, mask_tensor);
                    try_complete_pair(item.sequence_id, std::nullopt, std::move(mask_tensor), nullptr);

                } else {
                    lfs::core::Tensor decoded;
                    bool used_gpu = false;

                    if (decoder && item.is_original_jpeg) {
                        try {
                            decoded = load_image_hardware_decode(decoder, item.path, item.params);
                            used_gpu = true;
                        } catch (const std::exception&) {
                        }
                    }

                    if (!used_gpu) {
                        auto [img_data, width, height, channels] = lfs::core::load_image(
                            item.path, item.params.resize_factor, item.params.max_width);

                        if (!img_data)
                            throw std::runtime_error("Failed to decode image");

                        const size_t H = static_cast<size_t>(height);
                        const size_t W = static_cast<size_t>(width);
                        const size_t C = static_cast<size_t>(channels);

                        auto cpu_tensor = lfs::core::Tensor::from_blob(
                            img_data, lfs::core::TensorShape({H, W, C}),
                            lfs::core::Device::CPU, lfs::core::DataType::UInt8);

                        auto gpu_uint8 = cpu_tensor.to(lfs::core::Device::CUDA);
                        lfs::core::free_image(img_data);

                        decoded = lfs::core::Tensor::zeros(
                            lfs::core::TensorShape({C, H, W}),
                            lfs::core::Device::CUDA, lfs::core::DataType::Float32);

                        cuda::launch_uint8_hwc_to_float32_chw(
                            reinterpret_cast<const uint8_t*>(gpu_uint8.data_ptr()),
                            reinterpret_cast<float*>(decoded.data_ptr()),
                            H, W, C, nullptr);

                        if (const cudaError_t err = cudaDeviceSynchronize(); err != cudaSuccess) {
                            throw std::runtime_error(std::string("CUDA sync failed: ") + cudaGetErrorString(err));
                        }
                        gpu_uint8 = lfs::core::Tensor();
                    }

                    if (item.undistort) {
                        const auto scaled = lfs::core::scale_undistort_params(
                            *item.undistort,
                            static_cast<int>(decoded.shape()[2]),
                            static_cast<int>(decoded.shape()[1]));
                        decoded = lfs::core::undistort_image(decoded, scaled, nullptr);
                    }

                    if (nvcodec) {
                        try {
                            auto jpeg_bytes = nvcodec->encode_to_jpeg(
                                decoded, config_.cache_jpeg_quality, nullptr);
                            save_to_fs_cache(item.cache_key, jpeg_bytes);
                            put_in_jpeg_cache(item.cache_key,
                                              std::make_shared<std::vector<uint8_t>>(std::move(jpeg_bytes)));
                        } catch (const std::exception&) {
                        }
                    }

                    put_in_resident_gpu_cache(item.cache_key, decoded);
                    try_complete_pair(item.sequence_id, std::move(decoded), std::nullopt, nullptr);
                }

            } catch (const std::exception& e) {
                if (item.alpha_as_mask) {
                    LOG_WARN("[PipelinedImageLoader] Alpha-as-mask failed {}: {} - loading as RGB",
                             lfs::core::path_to_utf8(item.path), e.what());
                    try {
                        auto [img_data, width, height, channels] = lfs::core::load_image(
                            item.path, item.params.resize_factor, item.params.max_width);
                        if (!img_data)
                            throw std::runtime_error("RGB fallback also failed");

                        const size_t H = static_cast<size_t>(height);
                        const size_t W = static_cast<size_t>(width);
                        const size_t C = static_cast<size_t>(channels);

                        auto cpu_tensor = lfs::core::Tensor::from_blob(
                            img_data, lfs::core::TensorShape({H, W, C}),
                            lfs::core::Device::CPU, lfs::core::DataType::UInt8);
                        auto gpu_uint8 = cpu_tensor.to(lfs::core::Device::CUDA);
                        lfs::core::free_image(img_data);

                        auto decoded = lfs::core::Tensor::zeros(
                            lfs::core::TensorShape({C, H, W}),
                            lfs::core::Device::CUDA, lfs::core::DataType::Float32);
                        cuda::launch_uint8_hwc_to_float32_chw(
                            gpu_uint8.ptr<uint8_t>(), decoded.ptr<float>(), H, W, C, nullptr);
                        cudaStreamSynchronize(nullptr);

                        {
                            std::lock_guard<std::mutex> lock(pending_pairs_mutex_);
                            if (auto it = pending_pairs_.find(item.sequence_id); it != pending_pairs_.end()) {
                                it->second.mask_expected = false;
                            }
                        }
                        put_in_resident_gpu_cache(item.cache_key, decoded);
                        try_complete_pair(item.sequence_id, std::move(decoded), std::nullopt, nullptr);
                    } catch (const std::exception& e2) {
                        LOG_ERROR("[PipelinedImageLoader] RGB fallback also failed {}: {}",
                                  lfs::core::path_to_utf8(item.path), e2.what());
                        std::lock_guard<std::mutex> lock(pending_pairs_mutex_);
                        pending_pairs_.erase(item.sequence_id);
                        in_flight_.fetch_sub(1, std::memory_order_acq_rel);
                    }
                } else if (item.is_mask) {
                    LOG_WARN("[PipelinedImageLoader] Cold process mask error {}: {} - continuing without mask",
                             lfs::core::path_to_utf8(item.path), e.what());
                    // Mark mask as no longer expected so image can still be delivered
                    std::lock_guard<std::mutex> lock(pending_pairs_mutex_);
                    if (auto it = pending_pairs_.find(item.sequence_id); it != pending_pairs_.end()) {
                        it->second.mask_expected = false;
                        // If image is already ready, deliver it now
                        if (it->second.image.has_value()) {
                            output_queue_.push({item.sequence_id,
                                                std::move(*it->second.image),
                                                std::nullopt,
                                                it->second.stream});
                            pending_pairs_.erase(it);
                        }
                    }
                } else {
                    LOG_ERROR("[PipelinedImageLoader] Cold process error {}: {}",
                              lfs::core::path_to_utf8(item.path), e.what());
                    // Clean up pending_pairs_ to prevent memory leak
                    {
                        std::lock_guard<std::mutex> lock(pending_pairs_mutex_);
                        pending_pairs_.erase(item.sequence_id);
                    }
                    in_flight_.fetch_sub(1, std::memory_order_acq_rel);
                }
            }
        }
    }

} // namespace lfs::io
