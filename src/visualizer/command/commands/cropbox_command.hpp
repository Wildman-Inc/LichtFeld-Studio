/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "command/command.hpp"
#include "scene/scene.hpp"
#include <glm/glm.hpp>
#include <string>

namespace lfs::vis {
    class SceneManager;
}

namespace lfs::vis::command {

    // Scene graph cropbox state for undo/redo
    struct CropBoxState {
        glm::vec3 min{-1.0f};
        glm::vec3 max{1.0f};
        glm::mat4 local_transform{1.0f};
        bool inverse = false;
    };

    class CropBoxCommand : public Command {
    public:
        CropBoxCommand(SceneManager* scene_manager,
                       const std::string& cropbox_node_name,
                       const CropBoxState& old_state,
                       const CropBoxState& new_state);

        void undo() override;
        void redo() override;
        std::string getName() const override { return "CropBox"; }

    private:
        void applyState(const CropBoxState& state);

        SceneManager* scene_manager_;
        std::string cropbox_node_name_;
        CropBoxState old_state_;
        CropBoxState new_state_;
    };

} // namespace lfs::vis::command
