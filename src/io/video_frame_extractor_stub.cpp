/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "video_frame_extractor.hpp"

namespace lfs::io {

    VideoFrameExtractor::VideoFrameExtractor() : impl_(nullptr) {}

    VideoFrameExtractor::~VideoFrameExtractor() = default;

    bool VideoFrameExtractor::extract(const Params&, std::string& error) {
        error = "Video frame extraction is not available for HIP backend in this build";
        return false;
    }

} // namespace lfs::io
