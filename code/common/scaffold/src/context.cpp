#include "vkc/context.hpp"

#include "vkc/window.hpp"

namespace vkc {

// The order here is the dependency order, and it is the reason the three objects are
// built in three steps rather than one: the surface needs an instance to be created
// from, and the device cannot be chosen without a surface to ask about.
Context::Context(const Config& config, Window& window) : window_(&window) {
    instance_ = std::make_unique<Instance>(Instance::Config{
        .app_name = config.app_name,
        .enable_validation = config.enable_validation,
    });

    surface_ = window.create_surface(instance_->handle());

    device_ = std::make_unique<Device>(*instance_, surface_);
}

// Strict reverse order. The device owns the allocator and dies first; the surface was
// created from the instance and has to go before it; the instance goes last.
Context::~Context() {
    device_.reset();

    if (window_ != nullptr && instance_) {
        window_->destroy_surface(instance_->handle());
    }

    instance_.reset();
}

}  // namespace vkc
