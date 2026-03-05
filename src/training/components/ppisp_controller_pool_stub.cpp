/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "ppisp_controller_pool.hpp"
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

    PPISPControllerPool::PPISPControllerPool(const int num_cameras, const int total_iterations, Config config)
        : num_cameras_(num_cameras),
          total_iterations_(std::max(total_iterations, 1)),
          config_(config),
          current_lr_(config.lr),
          initial_lr_(config.lr) {}

    void PPISPControllerPool::allocate_buffers(const size_t max_h, const size_t max_w) {
        buf_h_ = std::max(buf_h_, max_h);
        buf_w_ = std::max(buf_w_, max_w);
    }

    lfs::core::Tensor PPISPControllerPool::predict(
        const int camera_idx,
        const lfs::core::Tensor& rendered_rgb,
        const float) {
        if (camera_idx < 0 || camera_idx >= num_cameras_) {
            throw std::runtime_error("PPISPControllerPool: camera_idx out of range");
        }
        last_predict_camera_ = camera_idx;
        const auto device = rendered_rgb.is_valid() ? rendered_rgb.device() : lfs::core::Device::CUDA;
        return lfs::core::Tensor::zeros({1, 9}, device);
    }

    void PPISPControllerPool::backward(const int camera_idx, const lfs::core::Tensor&) {
        if (camera_idx < 0 || camera_idx >= num_cameras_) {
            throw std::runtime_error("PPISPControllerPool: camera_idx out of range");
        }
    }

    void PPISPControllerPool::optimizer_step(const int camera_idx) {
        if (camera_idx < 0 || camera_idx >= num_cameras_) {
            throw std::runtime_error("PPISPControllerPool: camera_idx out of range");
        }
    }

    void PPISPControllerPool::zero_grad() {}

    void PPISPControllerPool::scheduler_step(const int camera_idx) {
        if (camera_idx < 0 || camera_idx >= num_cameras_) {
            throw std::runtime_error("PPISPControllerPool: camera_idx out of range");
        }

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

    void PPISPControllerPool::serialize(std::ostream& os) const {
        os.write(reinterpret_cast<const char*>(&CHECKPOINT_MAGIC), sizeof(CHECKPOINT_MAGIC));
        os.write(reinterpret_cast<const char*>(&CHECKPOINT_VERSION), sizeof(CHECKPOINT_VERSION));
        os.write(reinterpret_cast<const char*>(&num_cameras_), sizeof(num_cameras_));
        os.write(reinterpret_cast<const char*>(&total_iterations_), sizeof(total_iterations_));
        os.write(reinterpret_cast<const char*>(&config_), sizeof(config_));
        os.write(reinterpret_cast<const char*>(&step_), sizeof(step_));
        os.write(reinterpret_cast<const char*>(&current_lr_), sizeof(current_lr_));
        os.write(reinterpret_cast<const char*>(&initial_lr_), sizeof(initial_lr_));
    }

    void PPISPControllerPool::deserialize(std::istream& is) {
        uint32_t magic = 0;
        uint32_t version = 0;
        is.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        is.read(reinterpret_cast<char*>(&version), sizeof(version));
        if (magic != CHECKPOINT_MAGIC || version != CHECKPOINT_VERSION) {
            throw std::runtime_error("Invalid PPISPControllerPool checkpoint");
        }

        is.read(reinterpret_cast<char*>(&num_cameras_), sizeof(num_cameras_));
        is.read(reinterpret_cast<char*>(&total_iterations_), sizeof(total_iterations_));
        is.read(reinterpret_cast<char*>(&config_), sizeof(config_));
        is.read(reinterpret_cast<char*>(&step_), sizeof(step_));
        is.read(reinterpret_cast<char*>(&current_lr_), sizeof(current_lr_));
        is.read(reinterpret_cast<char*>(&initial_lr_), sizeof(initial_lr_));
    }

    void PPISPControllerPool::serialize_inference(std::ostream& os) const {
        os.write(reinterpret_cast<const char*>(&INFERENCE_MAGIC), sizeof(INFERENCE_MAGIC));
        os.write(reinterpret_cast<const char*>(&INFERENCE_VERSION), sizeof(INFERENCE_VERSION));
        os.write(reinterpret_cast<const char*>(&num_cameras_), sizeof(num_cameras_));
        os.write(reinterpret_cast<const char*>(&config_), sizeof(config_));
        os.write(reinterpret_cast<const char*>(&current_lr_), sizeof(current_lr_));
    }

    void PPISPControllerPool::deserialize_inference(std::istream& is) {
        uint32_t magic = 0;
        uint32_t version = 0;
        is.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        is.read(reinterpret_cast<char*>(&version), sizeof(version));
        if (magic != INFERENCE_MAGIC || version != INFERENCE_VERSION) {
            throw std::runtime_error("Invalid PPISPControllerPool inference file");
        }

        is.read(reinterpret_cast<char*>(&num_cameras_), sizeof(num_cameras_));
        is.read(reinterpret_cast<char*>(&config_), sizeof(config_));
        is.read(reinterpret_cast<char*>(&current_lr_), sizeof(current_lr_));
        initial_lr_ = current_lr_;
    }

    void PPISPControllerPool::adam_update(
        lfs::core::Tensor&,
        lfs::core::Tensor&,
        lfs::core::Tensor&,
        const lfs::core::Tensor&) {}

    void PPISPControllerPool::compute_bias_corrections(float& bc1_rcp, float& bc2_sqrt_rcp) const {
        const double bc1 = 1.0 - std::pow(config_.beta1, step_ + 1);
        const double bc2 = 1.0 - std::pow(config_.beta2, step_ + 1);
        bc1_rcp = static_cast<float>(1.0 / bc1);
        bc2_sqrt_rcp = static_cast<float>(1.0 / std::sqrt(bc2));
    }

} // namespace lfs::training
