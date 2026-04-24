/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/tensor.hpp"
#include "io/image_decode_format.hpp"

#include <filesystem>
#include <memory>
#include <vector>

namespace lfs::io {

    /**
     * @brief AMD AMF based JPEG image decoder.
     *
     * AMF is used only for JPEG/MJPEG decode. Output tensors follow the same
     * layout and normalization as NvCodecImageLoader.
     */
    class AmfImageLoader {
    public:
        struct Options {
            int device_id = 0;
            size_t decoder_pool_size = 1;
        };

        explicit AmfImageLoader(const Options& options);
        ~AmfImageLoader();

        AmfImageLoader(const AmfImageLoader&) = delete;
        AmfImageLoader& operator=(const AmfImageLoader&) = delete;

        lfs::core::Tensor load_image_gpu(
            const std::filesystem::path& path,
            int resize_factor = 1,
            int max_width = 0,
            void* cuda_stream = nullptr,
            DecodeFormat format = DecodeFormat::RGB);

        lfs::core::Tensor load_image_from_memory_gpu(
            const std::vector<uint8_t>& jpeg_data,
            int resize_factor = 1,
            int max_width = 0,
            void* cuda_stream = nullptr,
            DecodeFormat format = DecodeFormat::RGB);

        static bool is_available();

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;

        std::vector<uint8_t> read_file(const std::filesystem::path& path) const;
    };

} // namespace lfs::io
