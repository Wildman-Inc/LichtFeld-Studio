/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "io/amf_image_loader.hpp"

#include <stdexcept>

namespace lfs::io {

    namespace {
        [[noreturn]] void throw_disabled() {
            throw std::runtime_error("AMF image decode is disabled in this build");
        }
    } // namespace

    AmfImageLoader::AmfImageLoader(const Options&) {
        throw_disabled();
    }

    AmfImageLoader::~AmfImageLoader() = default;

    lfs::core::Tensor AmfImageLoader::load_image_gpu(
        const std::filesystem::path&, int, int, void*, DecodeFormat) {
        throw_disabled();
    }

    lfs::core::Tensor AmfImageLoader::load_image_from_memory_gpu(
        const std::vector<uint8_t>&, int, int, void*, DecodeFormat) {
        throw_disabled();
    }

    bool AmfImageLoader::is_available() {
        return false;
    }

    std::vector<uint8_t> AmfImageLoader::read_file(const std::filesystem::path&) const {
        throw_disabled();
    }

} // namespace lfs::io
