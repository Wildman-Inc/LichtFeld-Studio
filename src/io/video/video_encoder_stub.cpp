/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "video_encoder.hpp"

namespace lfs::io::video {

    namespace {
        constexpr const char* kUnsupportedMessage =
            "Video encoding is not available for HIP backend in this build";
    }

    class VideoEncoderImpl {
    public:
        [[nodiscard]] std::expected<void, std::string> open(
            const std::filesystem::path&,
            const VideoExportOptions&) {
            return std::unexpected(kUnsupportedMessage);
        }

        [[nodiscard]] std::expected<void, std::string> writeFrame(
            std::span<const uint8_t>,
            int,
            int) {
            return std::unexpected(kUnsupportedMessage);
        }

        [[nodiscard]] std::expected<void, std::string> writeFrameGpu(
            const void*,
            int,
            int,
            void*) {
            return std::unexpected(kUnsupportedMessage);
        }

        [[nodiscard]] std::expected<void, std::string> close() {
            return {};
        }

        [[nodiscard]] bool isOpen() const {
            return false;
        }
    };

    VideoEncoder::VideoEncoder() : impl_(std::make_unique<VideoEncoderImpl>()) {}
    VideoEncoder::~VideoEncoder() = default;
    VideoEncoder::VideoEncoder(VideoEncoder&&) noexcept = default;
    VideoEncoder& VideoEncoder::operator=(VideoEncoder&&) noexcept = default;

    std::expected<void, std::string> VideoEncoder::open(
        const std::filesystem::path& path, const VideoExportOptions& options) {
        return impl_->open(path, options);
    }

    std::expected<void, std::string> VideoEncoder::writeFrame(
        std::span<const uint8_t> rgba_data, const int width, const int height) {
        return impl_->writeFrame(rgba_data, width, height);
    }

    std::expected<void, std::string> VideoEncoder::writeFrameGpu(
        const void* const rgb_gpu_ptr, const int width, const int height, void* const stream) {
        return impl_->writeFrameGpu(rgb_gpu_ptr, width, height, stream);
    }

    std::expected<void, std::string> VideoEncoder::close() {
        return impl_->close();
    }

    bool VideoEncoder::isOpen() const {
        return impl_->isOpen();
    }

} // namespace lfs::io::video
