/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "video_player.hpp"

namespace lfs::io {

    class VideoPlayer::Impl {};

    VideoPlayer::VideoPlayer() = default;
    VideoPlayer::~VideoPlayer() = default;

    bool VideoPlayer::open(const std::filesystem::path&) {
        return false;
    }

    void VideoPlayer::close() {}

    bool VideoPlayer::isOpen() const {
        return false;
    }

    void VideoPlayer::play() {}
    void VideoPlayer::pause() {}
    void VideoPlayer::togglePlayPause() {}

    bool VideoPlayer::isPlaying() const {
        return false;
    }

    void VideoPlayer::seek(double) {}
    void VideoPlayer::seekFrame(int64_t) {}
    void VideoPlayer::stepForward() {}
    void VideoPlayer::stepBackward() {}

    bool VideoPlayer::update(double) {
        return false;
    }

    const uint8_t* VideoPlayer::currentFrameData() const {
        return nullptr;
    }

    int VideoPlayer::width() const {
        return 0;
    }

    int VideoPlayer::height() const {
        return 0;
    }

    double VideoPlayer::currentTime() const {
        return 0.0;
    }

    double VideoPlayer::duration() const {
        return 0.0;
    }

    int64_t VideoPlayer::currentFrameNumber() const {
        return 0;
    }

    int64_t VideoPlayer::totalFrames() const {
        return 0;
    }

    double VideoPlayer::fps() const {
        return 0.0;
    }

    std::vector<uint8_t> VideoPlayer::getThumbnail(double, int) {
        return {};
    }

} // namespace lfs::io
