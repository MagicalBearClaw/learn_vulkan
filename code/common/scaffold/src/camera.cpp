#include "vkc/camera.hpp"

#include <SDL3/SDL.h>

#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <cmath>

namespace vkc {

glm::vec3 Camera::front() const {
    const float yaw_radians = glm::radians(yaw);
    const float pitch_radians = glm::radians(pitch);
    return glm::normalize(glm::vec3{
        std::cos(yaw_radians) * std::cos(pitch_radians),
        std::sin(pitch_radians),
        std::sin(yaw_radians) * std::cos(pitch_radians),
    });
}

glm::vec3 Camera::right() const {
    return glm::normalize(glm::cross(front(), glm::vec3{0.0F, 1.0F, 0.0F}));
}

glm::mat4 Camera::view() const {
    return glm::lookAt(position, position + front(), glm::vec3{0.0F, 1.0F, 0.0F});
}

void Camera::handle_event(const SDL_Event& event, SDL_Window* window) {
    if (event.type == SDL_EVENT_MOUSE_BUTTON_DOWN &&
        event.button.button == SDL_BUTTON_RIGHT) {
        mouse_captured_ = !mouse_captured_;
        SDL_SetWindowRelativeMouseMode(window, mouse_captured_);
        return;
    }

    if (event.type == SDL_EVENT_MOUSE_MOTION && mouse_captured_) {
        yaw += event.motion.xrel * sensitivity;
        pitch -= event.motion.yrel * sensitivity;
        pitch = std::clamp(pitch, -89.0F, 89.0F);
    }
}

void Camera::update(float delta_time) {
    const bool* keys = SDL_GetKeyboardState(nullptr);

    float distance = speed * delta_time;
    if (keys[SDL_SCANCODE_LSHIFT] || keys[SDL_SCANCODE_RSHIFT]) {
        distance *= sprint_multiplier;
    }

    const glm::vec3 forward = front();
    const glm::vec3 sideways = right();

    if (keys[SDL_SCANCODE_W]) {
        position += forward * distance;
    }
    if (keys[SDL_SCANCODE_S]) {
        position -= forward * distance;
    }
    if (keys[SDL_SCANCODE_A]) {
        position -= sideways * distance;
    }
    if (keys[SDL_SCANCODE_D]) {
        position += sideways * distance;
    }
    if (keys[SDL_SCANCODE_SPACE]) {
        position.y += distance;
    }
    if (keys[SDL_SCANCODE_LCTRL]) {
        position.y -= distance;
    }
}

}  // namespace vkc
