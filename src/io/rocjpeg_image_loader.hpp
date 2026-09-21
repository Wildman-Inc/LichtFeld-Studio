/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/tensor.hpp"
#include <cstdint>
#include <memory>
#include <span>

namespace lfs::io {

    // JPEG-only decoder. Encoding, auxiliary maps, and unsupported inputs keep
    // their existing paths; availability does not imply nvImageCodec support.
    class RocJpegImageLoader {
    public:
        RocJpegImageLoader();
        ~RocJpegImageLoader();
        RocJpegImageLoader(const RocJpegImageLoader&) = delete;
        RocJpegImageLoader& operator=(const RocJpegImageLoader&) = delete;

        [[nodiscard]] bool available() const;
        // Returns an invalid tensor when hardware cannot decode this input.
        lfs::core::Tensor decode(std::span<const uint8_t> jpeg,
                                 int resize_factor, int max_width,
                                 bool output_uint8);

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace lfs::io
