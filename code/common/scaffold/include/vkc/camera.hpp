#pragma once

#include <glm/glm.hpp>

union SDL_Event;
struct SDL_Window;

namespace vkc {

// A first-person fly camera: WASD to move, mouse to look, space and ctrl for up and
// down, shift to sprint.
//
// Chapter 1.14 wrote every line of this and explained why it stores two angles rather
// than a matrix, why `right` is built against world up rather than the camera's own,
// and why the pitch is clamped just short of vertical.
//
// There is no Vulkan in here at all. A camera is a matrix, and the matrix is the
// inverse of where the camera is.
class Camera {
public:
    glm::vec3 position{0.0F, 0.0F, 4.0F};

    // Degrees. -90 yaw faces down -z, which is the conventional "forward".
    float yaw = -90.0F;
    float pitch = 0.0F;

    float speed = 4.0F;        // world units per second
    float sensitivity = 0.1F;  // degrees per pixel of mouse motion
    float sprint_multiplier = 4.0F;

    [[nodiscard]] glm::vec3 front() const;
    [[nodiscard]] glm::vec3 right() const;
    [[nodiscard]] glm::mat4 view() const;

    // Feed it every SDL event. Handles mouse look and the right-click capture toggle.
    void handle_event(const SDL_Event& event, SDL_Window* window);

    // Call once per frame with the frame time. Reads the keyboard state directly,
    // because movement wants "is this key held now", not "was it just pressed".
    void update(float delta_time);

    [[nodiscard]] bool mouse_captured() const noexcept { return mouse_captured_; }

private:
    bool mouse_captured_ = false;
};

}  // namespace vkc
