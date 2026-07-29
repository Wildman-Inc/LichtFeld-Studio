# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression checks for VkSplat viewport output lifetime hazards."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _read(rel_path: str) -> str:
    return (PROJECT_ROOT / rel_path).read_text(encoding="utf-8")


def test_vksplat_output_resize_waits_before_destroying_gui_sampled_images():
    source = _read("src/visualizer/rendering/vksplat_viewport_renderer.cpp")
    function_start = source.index("VksplatViewportRenderer::ensureOutputImages")
    function_end = source.index(
        "std::expected<void, std::string> VksplatViewportRenderer::ensureComposePipeline",
        function_start,
    )
    body = source[function_start:function_end]

    wait_pos = body.index("context.waitForSubmittedFrames()")
    destroy_color_pos = body.index("context.destroyExternalImage(slot.image)")
    destroy_depth_pos = body.index("context.destroyExternalImage(slot.depth_image)")

    assert "replacing_existing_output" in body
    assert wait_pos < destroy_color_pos
    assert wait_pos < destroy_depth_pos


def test_vksplat_external_inputs_are_retained_and_timeline_ordered():
    source = _read("src/visualizer/rendering/vksplat_viewport_renderer.cpp")
    function_start = source.index("VksplatViewportRenderer::prepareInputs")
    function_end = source.index(
        "void VksplatViewportRenderer::logVramBreakdownIfChanged",
        function_start,
    )
    body = source[function_start:function_end]

    assert "context.externalMemoryInteropEnabled()" in body
    assert "const bool can_bind_external" in body
    retain_pos = body.index("retired_input_storages_.emplace_back")
    wait_pos = body.index("waitForSplatInputStreams", retain_pos)
    signal_pos = body.index("timeline.cuda_semaphore.cudaSignal", wait_pos)
    vulkan_wait_pos = body.index("renderer_.addTimelineWait", signal_pos)
    refusal_pos = body.index("VkSplat refusing full input-copy fallback")
    assert retain_pos < wait_pos < signal_pos < vulkan_wait_pos < refusal_pos


def test_vksplat_initialization_failure_rolls_back_all_provisional_resources():
    source = _read("src/visualizer/rendering/vksplat_viewport_renderer.cpp")
    function_start = source.index("VksplatViewportRenderer::ensureInitialized")
    function_end = source.index(
        "VksplatViewportRenderer::nextRenderCompletionValue", function_start
    )
    body = source[function_start:function_end]

    provisional_pos = body.index("initialized_ = true")
    rollback_pos = body.index("auto rollback = ScopeExit")
    first_creation_pos = body.index("cudaStreamCreateWithFlags")
    commit_pos = body.index("initialization_committed = true")
    assert provisional_pos < rollback_pos < first_creation_pos < commit_pos

    rollback_body = body[rollback_pos:first_creation_pos]
    assert "if (initialization_committed)" in rollback_body
    assert "reset()" in rollback_body
    assert "VkSplat initialization rollback failed" in rollback_body

    setup_body = body[first_creation_pos:commit_pos]
    assert "createExternalTimelineSemaphore(0, render_complete_external_)" in setup_body
    assert "upload_timelines_[slot].initialize" in setup_body
    assert "overlay_upload_timelines_[slot].initialize" in setup_body
    assert "lod_engine_timeline_.initialize" in setup_body
    assert "selection_query_timeline_.initialize" in setup_body

    failure_positions = []
    search_from = 0
    while True:
        failure_pos = body.find("return std::unexpected", search_from)
        if failure_pos < 0:
            break
        failure_positions.append(failure_pos)
        search_from = failure_pos + 1
    assert failure_positions
    assert all(rollback_pos < failure_pos < commit_pos for failure_pos in failure_positions)
    assert body.count("context_ = &context") == 1
    assert body.index("context_ = &context") < provisional_pos


def test_cdna_image_interop_falls_back_before_importing_external_images():
    interop_header = _read("src/rendering/cuda_vulkan_interop.hpp")
    interop_host = _read("src/rendering/cuda_vulkan_interop.cpp")
    interop_device = _read("src/rendering/cuda_vulkan_interop.cu")
    service = _read("src/visualizer/rendering/viewport_interop_service.cpp")
    ui_texture = _read("src/visualizer/gui/vulkan_ui_texture.cpp")

    assert "cudaVulkanImageInteropSupported" in interop_header
    assert "hipDeviceAttributeImageSupport" in interop_host
    assert 'dlsym(RTLD_DEFAULT, "hipExternalMemoryGetMappedMipmappedArray")' in interop_host
    assert "__HIP_NO_IMAGE_SUPPORT" in interop_device
    assert "hipErrorNotSupported" in interop_device

    init_start = interop_host.index("bool CudaVulkanInterop::init(")
    init_end = interop_host.index("void CudaVulkanInterop::reset()", init_start)
    init_body = interop_host[init_start:init_end]
    owner_pos = init_body.index("NativeHandleOwner memory_handle")
    capability_pos = init_body.index("cudaVulkanImageInteropSupported()")
    assert owner_pos < capability_pos
    import_pos = init_body.index("cudaImportExternalMemory")
    assert capability_pos < import_pos
    assert "externalMemoryGetMappedMipmappedArrayFunction()" in init_body

    prepare_start = service.index("void ViewportInteropService::prepareChannel")
    prepare_end = service.index("void ViewportInteropService::prepareFrame", prepare_start)
    prepare_body = service[prepare_start:prepare_end]
    capability_pos = prepare_body.index("cudaVulkanImageInteropSupported()")
    create_pos = prepare_body.index("context.createExternalImage")
    assert capability_pos < create_pos
    assert "channel.disabled = true" in prepare_body[capability_pos:create_pos]
    assert "if (!lfs::rendering::cudaVulkanImageInteropSupported())" in ui_texture


def test_appearance_optimizer_writes_are_bracketed_by_reader_handshake():
    source = _read("src/training/trainer.cpp")
    train_start = source.index("lfs::Result<Trainer::StepDisposition> Trainer::train_step")
    section_start = source.index("if (in_controller_phase)", train_start)
    section_end = source.index("// Sparsity loss", section_start)
    body = source[section_start:section_end]

    for operation in (
        "ppisp_controller_pool_->optimizer_step",
        "bilateral_grid_->optimizer_step",
        "ppisp_->optimizer_step",
    ):
        operation_pos = body.index(operation)
        wait_pos = body.rfind("waitForModelReaders()", 0, operation_pos)
        record_pos = body.find("recordParamsReady()", operation_pos)
        assert wait_pos >= 0
        assert record_pos > operation_pos
        lock_pos = body.rfind("appearance_write_lock", 0, operation_pos)
        assert wait_pos > lock_pos >= 0
        assert lock_pos < operation_pos < record_pos


def test_cuda_vulkan_image_handoff_is_ordered_in_both_directions():
    header = _read("src/rendering/cuda_vulkan_interop.hpp")
    context_header = _read("src/visualizer/window/vulkan_context.hpp")
    service = _read("src/visualizer/rendering/viewport_interop_service.cpp")
    ui_texture = _read("src/visualizer/gui/vulkan_ui_texture.cpp")

    assert "bool wait(std::uint64_t value" in header
    assert "std::optional<TimelinePoint> wait" in context_header
    assert "std::optional<TimelinePoint> signal" in context_header

    prepare_start = service.index("void ViewportInteropService::prepareChannel")
    prepare_end = service.index("void ViewportInteropService::prepareFrame", prepare_start)
    prepare_body = service[prepare_start:prepare_end]
    wait_pos = prepare_body.index("target.interop.wait(target.timeline_value")
    copy_pos = prepare_body.index("target.interop.copyTensorToSurface", wait_pos)
    signal_pos = prepare_body.index("target.interop.signal(signal_value", copy_pos)
    acquire_pos = prepare_body.index(
        "ImmediateTransitionOptions::waitOn", signal_pos
    )
    assert wait_pos < copy_pos < signal_pos < acquire_pos

    assert "interop.wait(interop_timeline_value, upload_stream)" in ui_texture
    assert "interop.signal(signal_value, upload_stream)" in ui_texture


def test_vulkan_immediate_submit_reclaims_context_owned_work_asynchronously():
    header = _read("src/visualizer/window/vulkan_context.hpp")
    source = _read("src/visualizer/window/vulkan_context.cpp")

    pending_start = header.index("struct PendingImmediateSubmit")
    pending_end = header.index("\n        };", pending_start)
    pending_body = header[pending_start:pending_end]
    assert "VkCommandBuffer cmd" in pending_body
    assert "VkFence fence" in pending_body

    drain_start = source.index("bool VulkanContext::drainCompletedImmediateSubmits")
    drain_end = source.index("bool VulkanContext::transitionImageLayoutImmediate", drain_start)
    drain_body = source[drain_start:drain_end]
    query_pos = drain_body.index("vkGetFenceStatus")
    destroy_pos = drain_body.index("vkDestroyFence", query_pos)
    free_pos = drain_body.index("vkFreeCommandBuffers", destroy_pos)
    erase_pos = drain_body.index("pending_immediate_submits_.erase", free_pos)
    assert query_pos < destroy_pos < free_pos < erase_pos
    assert "status == VK_NOT_READY" in drain_body

    transition_start = drain_end
    transition_end = source.index("bool VulkanContext::createSwapchain", transition_start)
    transition_body = source[transition_start:transition_end]
    active_frame_pos = transition_body.index("if (frame_active_)")
    drain_pos = transition_body.index("drainCompletedImmediateSubmits()")
    allocate_pos = transition_body.index("vkAllocateCommandBuffers")
    submit_pos = transition_body.index("vkQueueSubmit")
    retain_pos = transition_body.index("pending_immediate_submits_.push_back", submit_pos)
    assert active_frame_pos < drain_pos < allocate_pos < submit_pos < retain_pos
    assert "kMaxPendingImmediateSubmits = 64" in transition_body
    assert "vkWaitForFences" not in transition_body

    wait_start = source.index("bool VulkanContext::waitForImmediateSubmits")
    wait_end = source.index("bool VulkanContext::deviceWaitIdle", wait_start)
    wait_body = source[wait_start:wait_end]
    assert wait_body.index("vkWaitForFences") < wait_body.index("vkDestroyFence")
    assert wait_body.index("vkDestroyFence") < wait_body.index("vkFreeCommandBuffers")
    assert wait_body.index("vkFreeCommandBuffers") < wait_body.index(
        "pending_immediate_submits_.clear"
    )


def test_cuda_surface_copy_is_stream_ordered_without_blocking_the_steady_path():
    header = _read("src/rendering/cuda_vulkan_interop.hpp")
    source = _read("src/rendering/cuda_vulkan_interop.cpp")
    service = _read("src/visualizer/rendering/viewport_interop_service.cpp")

    assert "copyTensorToSurface" in header
    assert "bool wait(std::uint64_t value" in header
    assert "bool signal(std::uint64_t value" in header

    copy_start = source.index("bool CudaVulkanInterop::copyTensorToSurface")
    copy_end = source.index("bool CudaVulkanInterop::wait", copy_start)
    copy_body = source[copy_start:copy_end]
    assert "upload_source_.sync_to_stream(stream)" in copy_body
    assert "launchCudaVulkanCopyTensorToSurface" in copy_body
    assert "cudaStreamSynchronize" not in copy_body

    prepare_start = service.index("void ViewportInteropService::prepareChannel")
    prepare_end = service.index("void ViewportInteropService::prepareFrame", prepare_start)
    prepare_body = service[prepare_start:prepare_end]
    wait_pos = prepare_body.index("target.interop.wait")
    copy_pos = prepare_body.index("target.interop.copyTensorToSurface", wait_pos)
    signal_pos = prepare_body.index("target.interop.signal", copy_pos)
    assert wait_pos < copy_pos < signal_pos


def test_scene_upload_uses_frame_slots_and_timeline_handoffs():
    service = _read("src/visualizer/rendering/viewport_interop_service.cpp")
    scene = _read("src/visualizer/scene/scene_manager.cpp")

    prepare_start = service.index("void ViewportInteropService::prepareChannel")
    prepare_end = service.index("void ViewportInteropService::prepareFrame", prepare_start)
    body = service[prepare_start:prepare_end]
    assert "const std::size_t frame_slot = context.currentFrameSlot()" in body
    assert "channel.targets.size() != context.framesInFlight()" in body
    assert "channel.targets.resize(context.framesInFlight())" in body
    cache_hit_pos = body.index("ViewportInteropAction::CacheHit")
    frame_wait_pos = body.index("context.waitForCurrentFrameSlot()")
    copy_pos = body.index("target.interop.copyTensorToSurface")
    assert cache_hit_pos < frame_wait_pos < copy_pos
    assert "target.interop.wait" in body
    assert "target.interop.signal" in body
    assert "cudaStreamSynchronize" not in body

    drain_start = scene.index("void SceneManager::drainGpuForTensorRelease()")
    drain_end = scene.index("bool SceneManager::resetToEmptyState", drain_start)
    drain_body = scene[drain_start:drain_end]
    release_pos = drain_body.index("viewportInterop().setSceneImage(nullptr")
    idle_pos = drain_body.index("vulkan_ctx->deviceWaitIdle()")
    assert release_pos < idle_pos


def test_cuda_vulkan_external_images_drain_work_before_destroy():
    service = _read("src/visualizer/rendering/viewport_interop_service.cpp")
    target_start = service.index("void destroy(VulkanContext& context)")
    target_end = service.index("};", target_start)
    target_body = service[target_start:target_end]

    idle_pos = target_body.index("context.waitForImmediateSubmits()")
    interop_reset_pos = target_body.index("\n            interop.reset();")
    image_destroy_pos = target_body.index("\n            context.destroyExternalImage(image);")
    assert idle_pos < interop_reset_pos < image_destroy_pos

    ui_texture = _read("src/visualizer/gui/vulkan_ui_texture.cpp")
    destroy_start = ui_texture.index("void destroyImage()")
    destroy_end = ui_texture.index("void reset()", destroy_start)
    destroy_body = ui_texture[destroy_start:destroy_end]
    assert "has_interop_resources" in destroy_body
    submitted_pos = destroy_body.index("context->waitForSubmittedFrames()")
    immediate_pos = destroy_body.index("context->waitForImmediateSubmits()")
    reset_pos = destroy_body.index("interop.reset()")
    assert submitted_pos < immediate_pos < reset_pos


def test_point_cloud_vulkan_viewport_capture_uses_readback():
    header = _read("src/visualizer/rendering/point_cloud_vulkan_renderer.hpp")
    source = _read("src/visualizer/rendering/point_cloud_vulkan_renderer.cpp")
    manager = _read("src/visualizer/rendering/rendering_manager_vulkan.cpp")

    assert "readOutputImage(" in header
    assert "vkCmdCopyImageToBuffer" in source
    assert "VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL" in source
    assert "PointCloudVulkanRenderer::readOutputImage" in source

    lazy_capture_start = manager.index("point_cloud_vulkan_renderer_->readOutputImage")
    lazy_capture_body = manager[lazy_capture_start - 500:lazy_capture_start + 500]
    assert "Failed to capture point-cloud Vulkan viewport image" in lazy_capture_body
    assert "[]() -> std::shared_ptr<lfs::core::Tensor> { return {}; }" not in lazy_capture_body


def test_python_render_view_uses_vulkan_preview_renderer():
    source = _read("src/python/lfs/py_rendering.cpp")
    function_start = source.index("std::optional<PyTensor> render_view(")
    function_end = source.index("std::optional<PyTensor> compute_screen_positions", function_start)
    body = source[function_start:function_end]

    assert "renderViewThreadSafe" in body
    assert "rendering::flipImageVertical" not in body
    assert "detectImageLayout" not in body
    assert "image->size(2) != 3" in body
    assert "(void)rotation;" not in body
    assert "(void)translation;" not in body

    helper_start = source.index("rendering_manager->renderPreviewImage")
    helper_body = source[helper_start - 500:helper_start + 500]
    assert "lfs::rendering::vFovToFocalLength" in helper_body
    assert "scene_manager" in helper_body


def test_vulkan_preview_render_view_uses_native_device_limits_not_16k_policy():
    header = _read("src/visualizer/rendering/rendering_manager.hpp")
    source = _read("src/visualizer/rendering/rendering_manager_viewport.cpp")
    context = _read("src/visualizer/window/vulkan_context.cpp")
    constants = _read("src/rendering/include/rendering/render_constants.hpp")

    assert "MAX_VIEWPORT_SIZE" not in constants
    assert "16384" not in source

    assert "VksplatViewportRenderer::OutputSlot::Preview" in source
    assert "format_properties.imageFormatProperties.maxExtent" in context
    assert "exceeds device-supported limit" in context
    assert "kMaxNativePreviewPixelStateBytes" in source
    assert "renderPreviewImageTiledWithState" in header
    assert "renderPreviewImageTiledWithState" in source
    assert "request.frame_view.intrinsics_override" in source
    assert "copyPreviewTileToOutput" not in source
    assert "readOutputImageIntoCpuHwc" in source
    assert "lfs::core::Tensor::empty" in source


def test_python_render_view_has_uint8_export_path():
    header = _read("src/python/lfs/py_rendering.hpp")
    source = _read("src/python/lfs/py_rendering.cpp")
    stub = _read("src/python/stubs/lichtfeld/__init__.pyi")

    assert "render_view_u8" in header
    assert "render_view_u8" in source
    assert "renderPreviewImageRgb8" in source
    assert "core::DataType::UInt8" in source
    assert "releasePreviewImageResources" in source
    assert "def render_view_u8" in stub


def test_vksplat_preview_export_releases_transient_resources():
    header = _read("src/visualizer/rendering/vksplat_viewport_renderer.hpp")
    source = _read("src/visualizer/rendering/vksplat_viewport_renderer.cpp")
    manager = _read("src/visualizer/rendering/rendering_manager_viewport.cpp")

    assert "releasePreviewResources" in header
    assert "releaseOutputSlot(OutputSlot::Preview)" in source
    assert "releasePrivateScratchBuffers()" in source
    assert "releaseSharedScratchArena()" in source
    assert "logVramBreakdownIfChanged(\"preview_release\")" in source
    assert "releasePreviewImageResources" in manager


def test_gaussian_video_export_uses_vulkan_preview_renderer():
    source = _read("src/visualizer/gui/async_task_manager.cpp")
    function_start = source.index("std::expected<lfs::core::Tensor, std::string> renderVideoExportFrame(")
    function_end = source.index("AsyncTaskManager::AsyncTaskManager", function_start)
    body = source[function_start:function_end]

    assert "Gaussian video export needs a Vulkan offscreen export path" not in body
    assert "rendering_manager.renderPreviewImage" in body
    assert "makeGaussianPreviewVideoFrame" in body
    assert "materializeGpuFrame" in body
