/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

namespace lfs::io {

    /**
     * @brief Output format for accelerated image decoding
     */
    enum class DecodeFormat {
        RGB,      // 3-channel RGB [C,H,W] or [H,W,C]
        Grayscale // 1-channel grayscale [H,W]
    };

} // namespace lfs::io
