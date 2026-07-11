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


def test_vksplat_synchronized_upload_without_timeline_uses_stable_copy():
    source = _read("src/visualizer/rendering/vksplat_viewport_renderer.cpp")
    function_start = source.index("VksplatViewportRenderer::prepareInputs")
    function_end = source.index(
        "void VksplatViewportRenderer::logVramBreakdownIfChanged",
        function_start,
    )
    body = source[function_start:function_end]

    assert "!synchronize_upload || cuda_timeline_interop_enabled_" in body
    assert "external_binding_synchronized && base_inputs_external" in body
    assert "synchronized_upload_without_timeline" in body
    assert "force_live_copy_refresh" in body
    assert "force_upload || input_snapshot_changed || force_live_copy_refresh" in body


def test_vksplat_initialization_failure_rolls_back_only_pending_resources():
    source = _read("src/visualizer/rendering/vksplat_viewport_renderer.cpp")
    function_start = source.index("VksplatViewportRenderer::ensureInitialized")
    function_end = source.index("// Exception-path safety", function_start)
    body = source[function_start:function_end]

    rollback_pos = body.index("auto initialization_rollback = ScopeExit")
    first_creation_pos = body.index("cudaStreamCreateWithFlags")
    commit_pos = body.index("const auto commit_timeline")
    assert rollback_pos < first_creation_pos < commit_pos

    rollback_helper_pos = body.index("const auto rollback_timeline")
    rollback_helper_body = body[rollback_helper_pos:rollback_pos]
    timeline_reset_pos = rollback_helper_body.index("timeline.cuda_semaphore.reset()")
    timeline_destroy_pos = rollback_helper_body.index(
        "context.destroyExternalSemaphore(timeline.vk_semaphore)"
    )
    assert timeline_reset_pos < timeline_destroy_pos

    rollback_end = body.index("\n        try {\n            if (!render_stream_)", rollback_pos)
    rollback_body = body[rollback_pos:rollback_end]
    completion_reset_pos = rollback_body.index("pending_render_complete_cuda.reset()")
    completion_destroy_pos = rollback_body.index(
        "context.destroyExternalSemaphore(pending_render_complete_external)"
    )
    assert completion_reset_pos < completion_destroy_pos
    assert "renderer_.cleanup()" in rollback_body
    assert "if (render_stream_created)" in rollback_body
    assert "cudaStreamDestroy(render_stream_)" in rollback_body

    setup_body = body[rollback_pos:commit_pos]
    assert "pending_render_complete_external" in setup_body
    assert "pending_upload_timelines" in setup_body
    assert "pending_overlay_upload_timelines" in setup_body
    assert "pending_lod_engine_timeline" in setup_body
    assert "pending_selection_query_timeline" in setup_body
    assert "createExternalTimelineSemaphore(0, render_complete_external_)" not in setup_body
    assert "for (auto& timeline : upload_timelines_)" not in setup_body
    assert "for (auto& timeline : overlay_upload_timelines_)" not in setup_body
    verify_device_pos = setup_body.index("verifyCudaMatchesVulkanDevice()")
    release_handle_pos = setup_body.index("releaseExternalSemaphoreNativeHandle")
    assert verify_device_pos < release_handle_pos

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

    initialized_pos = body.index("initialized_ = true", commit_pos)
    committed_pos = body.index("initialization_committed = true", initialized_pos)
    assert body.count("context_ = &context") == 1
    assert body.index("context_ = &context") > commit_pos
    assert body.index("cuda_timeline_interop_enabled_ =", commit_pos) > commit_pos
    assert commit_pos < initialized_pos < committed_pos


def test_cdna_image_interop_falls_back_before_importing_external_images():
    interop_header = _read("src/rendering/cuda_vulkan_interop.hpp")
    interop_host = _read("src/rendering/cuda_vulkan_interop.cpp")
    interop_device = _read("src/rendering/cuda_vulkan_interop.cu")
    gui = _read("src/visualizer/gui/gui_manager.cpp")
    ui_texture = _read("src/visualizer/gui/vulkan_ui_texture.cpp")

    assert "cudaVulkanImageInteropSupported" in interop_header
    assert "hipDeviceAttributeImageSupport" in interop_host
    assert 'dlsym(RTLD_DEFAULT, "hipExternalMemoryGetMappedMipmappedArray")' in interop_host
    assert "__HIP_NO_IMAGE_SUPPORT" in interop_device
    assert "hipErrorNotSupported" in interop_device

    init_start = interop_host.index("bool CudaVulkanInterop::initImpl")
    init_end = interop_host.index("void CudaVulkanInterop::reset()", init_start)
    init_body = interop_host[init_start:init_end]
    owner_pos = init_body.index("NativeHandleOwner memory_handle")
    capability_pos = init_body.index("cudaVulkanImageInteropSupported()")
    assert owner_pos < capability_pos
    assert "status = cudaExternalMemoryGetMappedMipmappedArray" not in init_body
    assert "mapped_mipmapped_array(&cuda_mip_" in init_body

    assert gui.count("if (!lfs::rendering::cudaVulkanImageInteropSupported())") == 3
    assert "if (!lfs::rendering::cudaVulkanImageInteropSupported())" in ui_texture


def test_appearance_optimizer_writes_are_bracketed_by_reader_handshake():
    source = _read("src/training/trainer.cpp")
    section_start = source.index("if (in_controller_phase)")
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
        assert "appearance_write_lock" in body[wait_pos - 400:record_pos]


def test_cuda_vulkan_image_handoff_is_ordered_in_both_directions():
    header = _read("src/rendering/cuda_vulkan_interop.hpp")
    context_header = _read("src/visualizer/window/vulkan_context.hpp")
    gui = _read("src/visualizer/gui/gui_manager.cpp")
    ui_texture = _read("src/visualizer/gui/vulkan_ui_texture.cpp")

    assert "bool wait(std::uint64_t value" in header
    assert "VkSemaphore signal_semaphore" in context_header

    implementation_namespace = gui.index("namespace {", gui.index("struct VulkanSceneInteropTarget {"))
    prepare_start = gui.index("bool prepareVulkanCudaImageWrite", implementation_namespace)
    prepare_end = gui.index("bool finishVulkanCudaImageWrite", prepare_start)
    prepare_body = gui[prepare_start:prepare_end]
    assert "target.semaphore.semaphore" in prepare_body
    assert "target.interop.wait(image_ready_value)" in prepare_body

    assert "interop.wait(image_ready_value)" in ui_texture
    assert "interop.signal(signal_value)" in ui_texture


def test_vulkan_immediate_submit_ticket_reclaims_context_owned_work_before_complete():
    header = _read("src/visualizer/window/vulkan_context.hpp")
    source = _read("src/visualizer/window/vulkan_context.cpp")

    ticket_start = header.index("class ImmediateSubmitTicket")
    ticket_end = header.index("\n        };", ticket_start)
    ticket_body = header[ticket_start:ticket_end]
    assert "std::shared_ptr<ImmediateSubmitState> state_" in ticket_body
    assert "VkFence" not in ticket_body

    pending_start = header.index("struct PendingImmediateSubmit")
    pending_end = header.index("\n        };", pending_start)
    pending_body = header[pending_start:pending_end]
    assert "VkCommandBuffer cmd" in pending_body
    assert "VkFence fence" in pending_body
    assert "std::shared_ptr<ImmediateSubmitState> completion" in pending_body

    declaration_start = header.index("transitionImageLayoutImmediate")
    declaration_end = header.index(");", declaration_start)
    declaration = header[declaration_start:declaration_end]
    signal_pos = declaration.index("std::uint64_t signal_value = 0")
    ticket_pos = declaration.index("ImmediateSubmitTicket* completion_ticket = nullptr")
    assert signal_pos < ticket_pos
    assert "pollImmediateSubmit(const ImmediateSubmitTicket& ticket)" in header

    complete_start = source.index("void VulkanContext::completeImmediateSubmit")
    complete_end = source.index(
        "VulkanContext::ImmediateSubmitPollResult VulkanContext::pollImmediateSubmit",
        complete_start,
    )
    complete_body = source[complete_start:complete_end]
    destroy_fence_pos = complete_body.index("vkDestroyFence")
    free_command_pos = complete_body.index("vkFreeCommandBuffers")
    publish_complete_pos = complete_body.index(
        ".status = ImmediateSubmitStatus::Complete"
    )
    assert destroy_fence_pos < publish_complete_pos
    assert free_command_pos < publish_complete_pos

    poll_start = complete_end
    poll_end = source.index("void VulkanContext::drainCompletedImmediateSubmits", poll_start)
    poll_body = source[poll_start:poll_end]
    assert "if (status == VK_NOT_READY)" in poll_body
    assert "if (status == VK_SUCCESS)" in poll_body
    reclaim_pos = poll_body.index("completeImmediateSubmit(*pending)")
    erase_pos = poll_body.index("pending_immediate_submits_.erase(pending)")
    complete_return_pos = poll_body.index("return state->poll_result", erase_pos)
    assert reclaim_pos < erase_pos < complete_return_pos
    assert ".status = ImmediateSubmitStatus::Error" in poll_body
    assert "vkGetFenceStatus(immediate submit ticket) failed" in poll_body

    drain_start = poll_end
    drain_end = source.index("bool VulkanContext::transitionImageLayoutImmediate", drain_start)
    drain_body = source[drain_start:drain_end]
    assert "status != VK_NOT_READY" in drain_body
    assert "vkGetFenceStatus(immediate submit drain) failed" in drain_body

    transition_start = drain_end
    transition_end = source.index("bool VulkanContext::createSwapchain", transition_start)
    transition_body = source[transition_start:transition_end]
    active_frame_pos = transition_body.index("if (frame_active_)")
    allocate_pos = transition_body.index("vkAllocateCommandBuffers")
    assert active_frame_pos < allocate_pos
    assert ".completion = std::move(completion)" in transition_body

    shutdown_start = source.index("void VulkanContext::shutdown()")
    shutdown_end = source.index("VkExtent2D VulkanContext::framebufferExtent", shutdown_start)
    shutdown_body = source[shutdown_start:shutdown_end]
    assert "completeImmediateSubmit(pending)" in shutdown_body


def test_cuda_surface_copy_event_is_polled_without_blocking_the_steady_path():
    header = _read("src/rendering/cuda_vulkan_interop.hpp")
    source = _read("src/rendering/cuda_vulkan_interop.cpp")

    assert "enum class SurfaceCopyStatus" in header
    assert "enqueueTensorToSurface" in header
    assert "pollSurfaceCopy(SurfaceCopyStatus& status)" in header

    enqueue_start = source.index("bool CudaVulkanInterop::enqueueTensorToSurface")
    enqueue_end = source.index("bool CudaVulkanInterop::pollSurfaceCopy", enqueue_start)
    enqueue_body = source[enqueue_start:enqueue_end]
    assert "surface copy is already pending" in enqueue_body
    create_pos = enqueue_body.index("cudaEventCreateWithFlags")
    record_pos = enqueue_body.index("cudaEventRecord")
    pending_pos = enqueue_body.rindex("surface_copy_status_ = SurfaceCopyStatus::Pending")
    assert "cudaEventDisableTiming" in enqueue_body
    assert create_pos < record_pos < pending_pos

    poll_start = enqueue_end
    poll_end = source.index("bool CudaVulkanInterop::wait", poll_start)
    poll_body = source[poll_start:poll_end]
    not_ready_pos = poll_body.index("query_status == cudaErrorNotReady")
    complete_pos = poll_body.index("surface_copy_status_ = SurfaceCopyStatus::Complete")
    assert not_ready_pos < complete_pos
    assert "cudaStreamSynchronize" not in poll_body

    reset_start = source.index("void CudaVulkanInterop::reset()")
    reset_end = source.index("bool CudaVulkanInterop::valid()", reset_start)
    reset_body = source[reset_start:reset_end]
    event_wait_pos = reset_body.index("cudaEventSynchronize")
    event_destroy_pos = reset_body.index("cudaEventDestroy")
    surface_destroy_pos = reset_body.index("cudaDestroySurfaceObject")
    assert event_wait_pos < event_destroy_pos < surface_destroy_pos


def test_timeline_disabled_scene_upload_uses_nonblocking_three_slot_ring():
    gui = _read("src/visualizer/gui/gui_manager.cpp")
    scene = _read("src/visualizer/scene/scene_manager.cpp")

    async_start = gui.index("void GuiManager::prepareVulkanSceneInteropAsync")
    async_end = gui.index("void GuiManager::prepareVulkanSceneInterop(", async_start)
    async_body = gui[async_start:async_end]
    assert "std::max<std::size_t>(3, context.framesInFlight() + 1)" in async_body
    assert "pollImmediateSubmit" in async_body
    assert "pollSurfaceCopy" in async_body
    assert "enqueueTensorToSurface" in async_body
    assert "vulkan_scene_published_slot_" in async_body
    assert "VulkanSceneAsyncState::VulkanWritePending" in async_body
    assert "VulkanSceneAsyncState::Writable" in async_body
    assert "VulkanSceneAsyncState::CopyPending" in async_body
    assert "VulkanSceneAsyncState::Ready" in async_body
    for blocking_call in (
        "waitForCurrentFrameSlot",
        "cudaStreamSynchronize",
        "vkWaitForFences",
        "deviceWaitIdle",
    ):
        assert blocking_call not in async_body

    dispatch_start = async_end
    dispatch_end = gui.index("void GuiManager::resetVulkanSplitRightInterop", dispatch_start)
    dispatch_body = gui[dispatch_start:dispatch_end]
    async_dispatch_pos = dispatch_body.index("prepareVulkanSceneInteropAsync(context)")
    legacy_wait_pos = dispatch_body.index("waitForCurrentFrameSlot")
    assert async_dispatch_pos < legacy_wait_pos

    needs_start = gui.index("bool GuiManager::needsAnimationFrame() const")
    needs_end = gui.index("bool GuiManager::isViewportExportLocked", needs_start)
    needs_body = gui[needs_start:needs_end]
    assert "vulkan_scene_async_interop_active_" in needs_body
    assert "VulkanSceneAsyncState::CopyPending" in needs_body
    assert "!latest_ready" in needs_body

    drain_start = scene.index("void SceneManager::drainGpuForTensorRelease()")
    drain_end = scene.index("void SceneManager::resetToEmptyState", drain_start)
    assert "gui_mgr->drainVulkanSceneInterop()" in scene[drain_start:drain_end]


def test_cuda_vulkan_external_images_drain_work_before_destroy():
    gui = _read("src/visualizer/gui/gui_manager.cpp")
    target_start = gui.index("void destroy(VulkanContext& context)")
    target_end = gui.index("};", target_start)
    target_body = gui[target_start:target_end]

    idle_pos = target_body.index("\n            (void)context.deviceWaitIdle();")
    interop_reset_pos = target_body.index("\n            interop.reset();")
    image_destroy_pos = target_body.index("\n            context.destroyExternalImage(image);")
    assert idle_pos < interop_reset_pos < image_destroy_pos

    ui_texture = _read("src/visualizer/gui/vulkan_ui_texture.cpp")
    destroy_start = ui_texture.index("void destroyImage()")
    destroy_end = ui_texture.index("void reset()", destroy_start)
    destroy_body = ui_texture[destroy_start:destroy_end]
    assert "has_interop_resources" in destroy_body
    assert "mode == Mode::CudaInterop" not in destroy_body
    assert destroy_body.index("context->deviceWaitIdle()") < destroy_body.index("interop.reset()")


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
