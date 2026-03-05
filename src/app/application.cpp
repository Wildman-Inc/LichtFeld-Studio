/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "app/application.hpp"
#include "config.h"
#include "app/splash_screen.hpp"
#include "control/command_api.hpp"
#include "core/checkpoint_format.hpp"
#include "core/cuda_version.hpp"
#include "core/event_bridge/command_center_bridge.hpp"
#include "core/events.hpp"
#include "core/image_loader.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "core/pinned_memory_allocator.hpp"
#include "core/scene.hpp"
#include "core/tensor.hpp"
#include "io/cache_image_loader.hpp"
#include "rendering/framebuffer_factory.hpp"
#include "training/trainer.hpp"
#include "training/training_setup.hpp"
#include "visualizer/visualizer.hpp"

#include "python/runner.hpp"
#include "visualizer/gui/panels/python_scripts_panel.hpp"
#include <algorithm>
#include <cstdlib>
#include <cuda_runtime.h>
#include <rasterization_api.h>

#ifdef WIN32
#include <windows.h>
#endif

namespace lfs::app {

    namespace {

        bool checkGpuRuntimeVersion();
        bool warmupGpuRuntime();

        int runHeadless(std::unique_ptr<lfs::core::param::TrainingParameters> params) {
            if (params->dataset.data_path.empty() && !params->resume_checkpoint) {
                LOG_ERROR("Headless mode requires --data-path or --resume");
                return 1;
            }

            if (!checkGpuRuntimeVersion()) {
                return 1;
            }
            lfs::core::bind_selected_gpu_device();
            cudaDeviceProp prop;
            const int selected_device = std::max(0, lfs::core::selected_gpu_device());
            if (cudaGetDeviceProperties(&prop, selected_device) == cudaSuccess) {
                LOG_INFO("Backend: {} | GPU[{}]: {} | arch {}.{}", LFS_GPU_BACKEND, selected_device, prop.name, prop.major, prop.minor);
            } else {
                LOG_INFO("Backend: {}", LFS_GPU_BACKEND);
            }
            lfs::event::CommandCenterBridge::instance().set(&lfs::training::CommandCenter::instance());

            {
                core::Scene scene;

                if (params->resume_checkpoint) {
                    LOG_INFO("Resuming from checkpoint: {}", core::path_to_utf8(*params->resume_checkpoint));

                    auto params_result = core::load_checkpoint_params(*params->resume_checkpoint);
                    if (!params_result) {
                        LOG_ERROR("Failed to load checkpoint params: {}", params_result.error());
                        return 1;
                    }
                    auto checkpoint_params = std::move(*params_result);

                    if (!params->dataset.data_path.empty())
                        checkpoint_params.dataset.data_path = params->dataset.data_path;
                    if (!params->dataset.output_path.empty())
                        checkpoint_params.dataset.output_path = params->dataset.output_path;

                    if (checkpoint_params.dataset.data_path.empty()) {
                        LOG_ERROR("Checkpoint has no dataset path and none provided via --data-path");
                        return 1;
                    }
                    if (!std::filesystem::exists(checkpoint_params.dataset.data_path)) {
                        LOG_ERROR("Dataset path does not exist: {}", core::path_to_utf8(checkpoint_params.dataset.data_path));
                        return 1;
                    }

                    if (const auto result = training::validateDatasetPath(checkpoint_params); !result) {
                        LOG_ERROR("Dataset validation failed: {}", result.error());
                        return 1;
                    }

                    if (const auto result = training::loadTrainingDataIntoScene(checkpoint_params, scene); !result) {
                        LOG_ERROR("Failed to load training data: {}", result.error());
                        return 1;
                    }

                    for (const auto* node : scene.getNodes()) {
                        if (node->type == core::NodeType::POINTCLOUD) {
                            scene.removeNode(node->name, false);
                            break;
                        }
                    }

                    auto splat_result = core::load_checkpoint_splat_data(*params->resume_checkpoint);
                    if (!splat_result) {
                        LOG_ERROR("Failed to load checkpoint splat data: {}", splat_result.error());
                        return 1;
                    }

                    auto splat_data = std::make_unique<core::SplatData>(std::move(*splat_result));
                    scene.addSplat("Model", std::move(splat_data), core::NULL_NODE);
                    scene.setTrainingModelNode("Model");

                    checkpoint_params.resume_checkpoint = *params->resume_checkpoint;

                    // Preserve runtime-only CLI flags when resuming headless training.
                    checkpoint_params.optimization.headless = params->optimization.headless;
                    checkpoint_params.optimization.auto_train = params->optimization.auto_train;
                    checkpoint_params.optimization.no_splash = params->optimization.no_splash;
                    checkpoint_params.optimization.no_interop = params->optimization.no_interop;
                    checkpoint_params.optimization.debug_python = params->optimization.debug_python;
                    checkpoint_params.optimization.debug_python_port = params->optimization.debug_python_port;

                    if (params->optimization.iterations != checkpoint_params.optimization.iterations)
                        checkpoint_params.optimization.iterations = params->optimization.iterations;

                    auto trainer = std::make_unique<training::Trainer>(scene);

                    if (!params->python_scripts.empty()) {
                        trainer->set_python_scripts(params->python_scripts);
                        vis::gui::panels::PythonScriptManagerState::getInstance().setScripts(params->python_scripts);
                    }

                    if (const auto result = trainer->initialize(checkpoint_params); !result) {
                        LOG_ERROR("Failed to initialize trainer: {}", result.error());
                        return 1;
                    }

                    const auto ckpt_result = trainer->load_checkpoint(*params->resume_checkpoint);
                    if (!ckpt_result) {
                        LOG_ERROR("Failed to restore checkpoint state: {}", ckpt_result.error());
                        return 1;
                    }
                    LOG_INFO("Resumed from iteration {}", *ckpt_result);

                    core::Tensor::trim_memory_pool();

                    if (const auto result = trainer->train(); !result) {
                        LOG_ERROR("Training error: {}", result.error());
                        if (!params->python_scripts.empty()) {
                            core::Tensor::shutdown_memory_pool();
                            core::PinnedMemoryAllocator::instance().shutdown();
                            python::finalize();
                            std::_Exit(1);
                        }
                        return 1;
                    }
                } else {
                    LOG_INFO("Starting headless training...");

                    if (const auto result = training::loadTrainingDataIntoScene(*params, scene); !result) {
                        LOG_ERROR("Failed to load training data: {}", result.error());
                        return 1;
                    }

                    if (const auto result = training::initializeTrainingModel(*params, scene); !result) {
                        LOG_ERROR("Failed to initialize model: {}", result.error());
                        return 1;
                    }

                    auto trainer = std::make_unique<training::Trainer>(scene);

                    if (!params->python_scripts.empty()) {
                        trainer->set_python_scripts(params->python_scripts);
                        vis::gui::panels::PythonScriptManagerState::getInstance().setScripts(params->python_scripts);
                    }

                    if (const auto result = trainer->initialize(*params); !result) {
                        LOG_ERROR("Failed to initialize trainer: {}", result.error());
                        return 1;
                    }

                    core::Tensor::trim_memory_pool();

                    if (const auto result = trainer->train(); !result) {
                        LOG_ERROR("Training error: {}", result.error());
                        if (!params->python_scripts.empty()) {
                            core::Tensor::shutdown_memory_pool();
                            core::PinnedMemoryAllocator::instance().shutdown();
                            python::finalize();
                            std::_Exit(1);
                        }
                        return 1;
                    }
                }

                LOG_INFO("Headless training completed");
            }

            core::Tensor::shutdown_memory_pool();
            core::PinnedMemoryAllocator::instance().shutdown();

            if (!params->python_scripts.empty()) {
                python::finalize();
                std::_Exit(0);
            }
            return 0;
        }

        bool checkGpuRuntimeVersion() {
#if LFS_USE_HIP
            int runtime_version = 0;
            if (cudaRuntimeGetVersion(&runtime_version) == cudaSuccess) {
                LOG_INFO("HIP runtime version: {}.{}", runtime_version / 1000, (runtime_version % 1000) / 10);
            } else {
                LOG_WARN("Failed to query HIP runtime version");
            }

            const auto probe = lfs::core::ensure_gpu_runtime_ready();
            if (!probe.available) {
                LOG_ERROR("HIP device query failed: {}", probe.error);
                LOG_ERROR("Verify ROCm install and runtime visibility (hipInfo / rocminfo).");
                return false;
            }
            if (!lfs::core::bind_selected_gpu_device()) {
                LOG_ERROR("Failed to bind HIP device {}", probe.selected_device);
                return false;
            }
            return true;
#else
            const auto info = lfs::core::check_cuda_version();
            if (info.query_failed) {
                LOG_WARN("Failed to query CUDA driver version");
                return true;
            }

            LOG_INFO("CUDA driver version: {}.{}", info.major, info.minor);
            if (!info.supported) {
                LOG_WARN("CUDA {}.{} unsupported. Requires 12.8+ (driver 570+)", info.major, info.minor);
                return false;
            }
            const auto probe = lfs::core::ensure_gpu_runtime_ready();
            if (!probe.available) {
                LOG_ERROR("CUDA device query failed: {}", probe.error);
                return false;
            }
            if (!lfs::core::bind_selected_gpu_device()) {
                LOG_ERROR("Failed to bind CUDA device {}", probe.selected_device);
                return false;
            }
            return true;
#endif
        }

        bool warmupGpuRuntime() {
            if (!checkGpuRuntimeVersion()) {
                return false;
            }
            if (!lfs::core::bind_selected_gpu_device()) {
                LOG_ERROR("Failed to activate selected GPU device");
                return false;
            }

            cudaDeviceProp prop;
            const int selected_device = std::max(0, lfs::core::selected_gpu_device());
            if (cudaGetDeviceProperties(&prop, selected_device) == cudaSuccess) {
                LOG_INFO("GPU[{}]: {} (arch {}.{}, {} MB)", selected_device, prop.name, prop.major, prop.minor,
                         prop.totalGlobalMem / (1024 * 1024));
            }

            LOG_INFO("Initializing {} backend...", LFS_GPU_BACKEND);
            fast_lfs::rasterization::warmup_kernels();
            return true;
        }

        int runGui(std::unique_ptr<lfs::core::param::TrainingParameters> params) {
            if (params->optimization.no_interop) {
                LOG_INFO("GPU-OpenGL interop disabled");
                lfs::rendering::disableInterop();
            }

            if (!params->python_scripts.empty()) {
                vis::gui::panels::PythonScriptManagerState::getInstance().setScripts(params->python_scripts);
            }

            if (params->optimization.no_splash) {
                if (!warmupGpuRuntime()) {
                    return 1;
                }
            } else {
                const int warmup_result = SplashScreen::runWithDelay([]() {
                    return warmupGpuRuntime() ? 0 : 1;
                });
                if (warmup_result != 0) {
                    return warmup_result;
                }
            }

            lfs::event::CommandCenterBridge::instance().set(&lfs::training::CommandCenter::instance());

            auto viewer = vis::Visualizer::create({
                .title = "LichtFeld Studio",
                .width = 1280,
                .height = 720,
                .antialiasing = false,
                .enable_cuda_interop = true,
                .gut = params->optimization.gut,
            });

            viewer->setParameters(*params);

            for (const auto& vp : params->view_paths) {
                if (!std::filesystem::exists(vp)) {
                    LOG_ERROR("File not found: {}", lfs::core::path_to_utf8(vp));
                    return 1;
                }
            }
            if (!params->dataset.data_path.empty() && !std::filesystem::exists(params->dataset.data_path)) {
                LOG_ERROR("Dataset not found: {}", lfs::core::path_to_utf8(params->dataset.data_path));
                return 1;
            }

            if (params->import_cameras_path) {
                LOG_INFO("Importing COLMAP cameras: {}", lfs::core::path_to_utf8(*params->import_cameras_path));
                lfs::core::events::cmd::ImportColmapCameras{.sparse_path = *params->import_cameras_path}.emit();
            } else if (params->resume_checkpoint) {
                LOG_INFO("Loading checkpoint: {}", lfs::core::path_to_utf8(*params->resume_checkpoint));
                if (const auto result = viewer->loadCheckpointForTraining(*params->resume_checkpoint); !result) {
                    LOG_ERROR("Failed to load checkpoint: {}", result.error());
                    return 1;
                }
            }

            viewer->run();
            viewer.reset();

            core::Tensor::shutdown_memory_pool();
            core::PinnedMemoryAllocator::instance().shutdown();

            python::finalize();
            std::_Exit(0);
        }

#ifdef WIN32
        void hideConsoleWindow() {
            HWND hwnd = GetConsoleWindow();
            Sleep(1);
            HWND owner = GetWindow(hwnd, GW_OWNER);
            DWORD processId;
            GetWindowThreadProcessId(hwnd, &processId);

            if (GetCurrentProcessId() == processId) {
                ShowWindow(owner ? owner : hwnd, SW_HIDE);
            }
        }
#endif

    } // namespace

    int Application::run(std::unique_ptr<lfs::core::param::TrainingParameters> params) {
        lfs::core::set_image_loader([](const lfs::core::ImageLoadParams& p) {
            return lfs::io::CacheLoader::getInstance().load_cached_image(
                p.path, {.resize_factor = p.resize_factor, .max_width = p.max_width, .cuda_stream = p.stream});
        });

        if (params->optimization.headless) {
            return runHeadless(std::move(params));
        }

#ifdef WIN32
        hideConsoleWindow();
#endif

        return runGui(std::move(params));
    }

} // namespace lfs::app
