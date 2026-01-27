/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "tool_base.hpp"
#include <glm/glm.hpp>
#include <vector>

namespace lfs::vis::tools {

    class AlignTool : public ToolBase {
    public:
        AlignTool();
        ~AlignTool() override = default;

        std::string_view getName() const override { return "Align Tool"; }
        std::string_view getDescription() const override { return "3-point alignment to world axes"; }

        bool initialize(const ToolContext& ctx) override;
        void shutdown() override;
        void update(const ToolContext& ctx) override;
        void renderUI(const lfs::vis::gui::UIContext& ui_ctx, bool* p_open) override;

        bool handleMouseButton(int button, int action, double x, double y, const ToolContext& ctx);

    protected:
        void onEnabledChanged(bool enabled) override;

    private:
        std::vector<glm::vec3> picked_points_;
        const ToolContext* tool_context_ = nullptr;

        void reset();
        void applyAlignment(const ToolContext& ctx);
        glm::vec3 unprojectScreenPoint(double x, double y, const ToolContext& ctx);
    };

} // namespace lfs::vis::tools
