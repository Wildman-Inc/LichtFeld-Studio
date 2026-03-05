/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "ppisp_controller.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <istream>
#include <ostream>
#include <stdexcept>

namespace lfs::training {

    namespace {
        constexpr uint32_t CHECKPOINT_MAGIC = 0x4C465043;
        constexpr uint32_t CHECKPOINT_VERSION = 1;
        constexpr uint32_t INFERENCE_MAGIC = 0x4C464349;
        constexpr uint32_t INFERENCE_VERSION = 1;
    } // namespace

    size_t PPISPController::shared_buf_h_ = 0;
    size_t PPISPController::shared_buf_w_ = 0;
    lfs::core::Tensor PPISPController::shared_buf_conv1_;
    lfs::core::Tensor PPISPController::shared_buf_pool_;
    lfs::core::Tensor PPISPController::shared_buf_conv2_;
    lfs::core::Tensor PPISPController::shared_buf_conv3_;
    lfs::core::Tensor PPISPController::shared_buf_pool2_;

    PPISPController::PPISPController(const int total_iterations, Config config)
        : config_(config),
          current_lr_(config.lr),
          initial_lr_(config.lr),
          total_iterations_(std::max(total_iterations, 1)) {}

    void PPISPController::preallocate_shared_buffers(const size_t max_H, const size_t max_W) {
        shared_buf_h_ = std::max(shared_buf_h_, max_H);
        shared_buf_w_ = std::max(shared_buf_w_, max_W);
    }

    lfs::core::Tensor PPISPController::predict(const lfs::core::Tensor& rendered_rgb, const float) {
        const auto device = rendered_rgb.is_valid() ? rendered_rgb.device() : lfs::core::Device::CUDA;
        return lfs::core::Tensor::zeros({1, 9}, device);
    }

    void PPISPController::backward(const lfs::core::Tensor&) {}

    void PPISPController::optimizer_step() {}

    void PPISPController::zero_grad() {}

    void PPISPController::scheduler_step() {
        ++step_;
        if (step_ <= config_.warmup_steps) {
            const double progress = static_cast<double>(step_) / std::max(config_.warmup_steps, 1);
            const double scale = config_.warmup_start_factor + (1.0 - config_.warmup_start_factor) * progress;
            current_lr_ = initial_lr_ * scale;
            return;
        }

        const int decay_iters = std::max(total_iterations_ - config_.warmup_steps, 1);
        const double gamma = std::pow(config_.final_lr_factor, 1.0 / static_cast<double>(decay_iters));
        current_lr_ = initial_lr_ * std::pow(gamma, step_ - config_.warmup_steps);
    }

    void PPISPController::serialize(std::ostream& os) const {
        os.write(reinterpret_cast<const char*>(&CHECKPOINT_MAGIC), sizeof(CHECKPOINT_MAGIC));
        os.write(reinterpret_cast<const char*>(&CHECKPOINT_VERSION), sizeof(CHECKPOINT_VERSION));
        os.write(reinterpret_cast<const char*>(&config_), sizeof(config_));
        os.write(reinterpret_cast<const char*>(&step_), sizeof(step_));
        os.write(reinterpret_cast<const char*>(&current_lr_), sizeof(current_lr_));
        os.write(reinterpret_cast<const char*>(&initial_lr_), sizeof(initial_lr_));
        os.write(reinterpret_cast<const char*>(&total_iterations_), sizeof(total_iterations_));
    }

    void PPISPController::deserialize(std::istream& is) {
        uint32_t magic = 0;
        uint32_t version = 0;
        is.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        is.read(reinterpret_cast<char*>(&version), sizeof(version));
        if (magic != CHECKPOINT_MAGIC || version != CHECKPOINT_VERSION) {
            throw std::runtime_error("Invalid PPISPController checkpoint");
        }

        is.read(reinterpret_cast<char*>(&config_), sizeof(config_));
        is.read(reinterpret_cast<char*>(&step_), sizeof(step_));
        is.read(reinterpret_cast<char*>(&current_lr_), sizeof(current_lr_));
        is.read(reinterpret_cast<char*>(&initial_lr_), sizeof(initial_lr_));
        is.read(reinterpret_cast<char*>(&total_iterations_), sizeof(total_iterations_));
    }

    void PPISPController::serialize_inference(std::ostream& os) const {
        os.write(reinterpret_cast<const char*>(&INFERENCE_MAGIC), sizeof(INFERENCE_MAGIC));
        os.write(reinterpret_cast<const char*>(&INFERENCE_VERSION), sizeof(INFERENCE_VERSION));
        os.write(reinterpret_cast<const char*>(&config_), sizeof(config_));
        os.write(reinterpret_cast<const char*>(&current_lr_), sizeof(current_lr_));
    }

    void PPISPController::deserialize_inference(std::istream& is) {
        uint32_t magic = 0;
        uint32_t version = 0;
        is.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        is.read(reinterpret_cast<char*>(&version), sizeof(version));
        if (magic != INFERENCE_MAGIC || version != INFERENCE_VERSION) {
            throw std::runtime_error("Invalid PPISPController inference file");
        }

        is.read(reinterpret_cast<char*>(&config_), sizeof(config_));
        is.read(reinterpret_cast<char*>(&current_lr_), sizeof(current_lr_));
        initial_lr_ = current_lr_;
    }

    void PPISPController::adam_update(
        lfs::core::Tensor&,
        lfs::core::Tensor&,
        lfs::core::Tensor&,
        const lfs::core::Tensor&) {}

} // namespace lfs::training
