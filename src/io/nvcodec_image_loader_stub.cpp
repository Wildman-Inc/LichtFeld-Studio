/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "io/nvcodec_image_loader.hpp"
#include <stdexcept>

namespace lfs::io {

    struct NvCodecImageLoader::Impl {
    };

    namespace {
        [[noreturn]] void throw_disabled() {
            throw std::runtime_error("nvImageCodec is disabled in this build; use CPU image decode path");
        }
    } // namespace

    NvCodecImageLoader::NvCodecImageLoader(const Options&) {
        throw_disabled();
    }

    NvCodecImageLoader::~NvCodecImageLoader() = default;

    lfs::core::Tensor NvCodecImageLoader::load_image_gpu(
        const std::filesystem::path&, int, int, void*, DecodeFormat) {
        throw_disabled();
    }

    lfs::core::Tensor NvCodecImageLoader::load_image_from_memory_gpu(
        const std::vector<uint8_t>&, int, int, void*, DecodeFormat) {
        throw_disabled();
    }

    std::vector<lfs::core::Tensor> NvCodecImageLoader::load_images_batch_gpu(
        const std::vector<std::filesystem::path>&, int, int) {
        throw_disabled();
    }

    std::vector<lfs::core::Tensor> NvCodecImageLoader::batch_decode_from_memory(
        const std::vector<std::vector<uint8_t>>&, void*) {
        throw_disabled();
    }

    std::vector<lfs::core::Tensor> NvCodecImageLoader::batch_decode_from_spans(
        const std::vector<std::pair<const uint8_t*, size_t>>&, void*) {
        throw_disabled();
    }

    std::vector<uint8_t> NvCodecImageLoader::encode_to_jpeg(
        const lfs::core::Tensor&, int, void*) {
        throw_disabled();
    }

    std::vector<uint8_t> NvCodecImageLoader::encode_grayscale_to_jpeg(
        const lfs::core::Tensor&, int, void*) {
        throw_disabled();
    }

    bool NvCodecImageLoader::is_available() {
        return false;
    }

    std::vector<uint8_t> NvCodecImageLoader::read_file(const std::filesystem::path&) {
        throw_disabled();
    }

} // namespace lfs::io
