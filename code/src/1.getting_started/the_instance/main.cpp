// 1.2 The Instance
//
// Adds to 1.1: a VkInstance, the validation layers, and a debug messenger.
//
// The instance is Vulkan's root object. It is where you declare which API version you
// intend to use, which layers you want wrapped around your calls, and which
// instance-level extensions you need. Everything else in Vulkan is created from it,
// directly or indirectly.

#include <vkc/capture.hpp>
#include <vkc/check.hpp>

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <format>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

constexpr const char* kValidationLayer = "VK_LAYER_KHRONOS_validation";

// The validation layers are the single most valuable thing in the Vulkan ecosystem.
// Vulkan itself does almost no error checking -- that is why it is fast -- so without
// them a mistake produces a blank screen, a crash, or nothing at all. With them you
// get a paragraph telling you which struct member was wrong and which line of the
// specification you violated.
bool layer_available(std::string_view name) {
    uint32_t count = 0;
    VK_CHECK(vkEnumerateInstanceLayerProperties(&count, nullptr));
    std::vector<VkLayerProperties> layers(count);
    VK_CHECK(vkEnumerateInstanceLayerProperties(&count, layers.data()));

    return std::ranges::any_of(layers, [name](const VkLayerProperties& layer) {
        return name == layer.layerName;
    });
}

bool instance_extension_available(std::string_view name) {
    uint32_t count = 0;
    VK_CHECK(vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr));
    std::vector<VkExtensionProperties> extensions(count);
    VK_CHECK(vkEnumerateInstanceExtensionProperties(nullptr, &count, extensions.data()));

    return std::ranges::any_of(extensions, [name](const VkExtensionProperties& ext) {
        return name == ext.extensionName;
    });
}

// Where validation messages arrive. Vulkan calls this from whichever thread provoked
// the message, so anything it touches must be safe to touch from anywhere; logging is.
VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT /*types*/,
    const VkDebugUtilsMessengerCallbackDataEXT* data, void* /*user_data*/) {
    if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
        spdlog::error("[vulkan] {}", data->pMessage);
    } else if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        spdlog::warn("[vulkan] {}", data->pMessage);
    } else {
        spdlog::info("[vulkan] {}", data->pMessage);
    }
    // VK_FALSE means "carry on with the call". VK_TRUE aborts it, and is only useful
    // to people developing the layers themselves.
    return VK_FALSE;
}

VkDebugUtilsMessengerCreateInfoEXT debug_messenger_info() {
    return VkDebugUtilsMessengerCreateInfoEXT{
        .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
        .pNext = nullptr,
        .flags = 0,
        // Info and verbose are extremely chatty. Warnings and errors are the ones
        // that mean you have done something wrong.
        .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
        .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                       VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                       VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
        .pfnUserCallback = debug_callback,
        .pUserData = nullptr,
    };
}

class InstanceApp {
public:
    explicit InstanceApp(const vkc::Args& args) : args_(args) {
        init_window();
        init_instance();
    }

    ~InstanceApp() {
        if (debug_messenger_ != VK_NULL_HANDLE) {
            vkDestroyDebugUtilsMessengerEXT(instance_, debug_messenger_, nullptr);
        }
        if (instance_ != VK_NULL_HANDLE) {
            vkDestroyInstance(instance_, nullptr);
        }
        if (window_ != nullptr) {
            SDL_DestroyWindow(window_);
        }
        SDL_Quit();
    }

    InstanceApp(const InstanceApp&) = delete;
    InstanceApp& operator=(const InstanceApp&) = delete;
    InstanceApp(InstanceApp&&) = delete;
    InstanceApp& operator=(InstanceApp&&) = delete;

    int run() {
        report();

        uint64_t frames = 0;
        bool running = true;
        while (running) {
            SDL_Event event;
            while (SDL_PollEvent(&event)) {
                if (event.type == SDL_EVENT_QUIT ||
                    event.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED ||
                    (event.type == SDL_EVENT_KEY_DOWN && event.key.key == SDLK_ESCAPE)) {
                    running = false;
                }
            }
            ++frames;
            if (args_.frame_limit != 0 && frames >= args_.frame_limit) {
                running = false;
            }
        }
        return 0;
    }

private:
    void init_window() {
        if (!SDL_Init(SDL_INIT_VIDEO)) {
            throw std::runtime_error(std::format("SDL_Init failed: {}", SDL_GetError()));
        }
        window_ = SDL_CreateWindow("LearnVulkan - The Instance",
                                   static_cast<int>(args_.width),
                                   static_cast<int>(args_.height),
                                   SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);
        if (window_ == nullptr) {
            throw std::runtime_error(
                std::format("SDL_CreateWindow failed: {}", SDL_GetError()));
        }
    }

    void init_instance() {
        // volk finds the Vulkan loader and fetches the three or four entry points that
        // exist before an instance does. Every other vk* function is a null pointer
        // until this has run.
        VK_CHECK(volkInitialize());

        const VkApplicationInfo app_info{
            .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
            .pNext = nullptr,
            .pApplicationName = "LearnVulkan",
            .applicationVersion = VK_MAKE_VERSION(0, 1, 0),
            .pEngineName = "LearnVulkan",
            .engineVersion = VK_MAKE_VERSION(0, 1, 0),
            // The version you promise to obey. Asking for 1.3 is what makes dynamic
            // rendering and synchronization2 available as core features later.
            .apiVersion = VK_API_VERSION_1_3,
        };

        // SDL knows which surface extension this platform needs -- VK_KHR_surface
        // plus VK_KHR_wayland_surface, VK_KHR_win32_surface, and so on. Asking it
        // rather than hard-coding is what keeps this file platform-independent.
        uint32_t sdl_extension_count = 0;
        const char* const* sdl_extensions =
            SDL_Vulkan_GetInstanceExtensions(&sdl_extension_count);
        if (sdl_extensions == nullptr) {
            throw std::runtime_error(std::format(
                "SDL_Vulkan_GetInstanceExtensions failed: {}", SDL_GetError()));
        }

        std::vector<const char*> extensions(sdl_extensions,
                                            sdl_extensions + sdl_extension_count);

        validation_enabled_ = args_.validation && layer_available(kValidationLayer);
        if (args_.validation && !validation_enabled_) {
            spdlog::warn(
                "Validation layers requested but {} is not installed. Install your "
                "distribution's vulkan-validation-layers package (or the Vulkan SDK) "
                "before going any further: developing without them is developing "
                "blind.",
                kValidationLayer);
        }

        debug_utils_enabled_ =
            instance_extension_available(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        if (debug_utils_enabled_) {
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }

        std::vector<const char*> layers;
        if (validation_enabled_) {
            layers.push_back(kValidationLayer);
        }

        // Chaining the messenger description onto pNext gets validation coverage of
        // vkCreateInstance itself. A real messenger cannot exist yet -- it is created
        // from the instance we are in the middle of creating -- so this is the only
        // way to see errors in this call.
        const VkDebugUtilsMessengerCreateInfoEXT messenger_info = debug_messenger_info();
        const bool debug_at_creation = validation_enabled_ && debug_utils_enabled_;

        const VkInstanceCreateInfo create_info{
            .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            .pNext = debug_at_creation ? &messenger_info : nullptr,
            .flags = 0,
            .pApplicationInfo = &app_info,
            .enabledLayerCount = static_cast<uint32_t>(layers.size()),
            .ppEnabledLayerNames = layers.data(),
            .enabledExtensionCount = static_cast<uint32_t>(extensions.size()),
            .ppEnabledExtensionNames = extensions.data(),
        };

        VK_CHECK(vkCreateInstance(&create_info, nullptr, &instance_));

        // Now that an instance exists, volk can load the ~200 instance-level entry
        // points. vkCreateDebugUtilsMessengerEXT is one of them, so this has to happen
        // before the next line.
        volkLoadInstanceOnly(instance_);

        if (debug_at_creation) {
            VK_CHECK(vkCreateDebugUtilsMessengerEXT(instance_, &messenger_info, nullptr,
                                                    &debug_messenger_));
        }

        enabled_extensions_ = std::move(extensions);
    }

    void report() const {
        uint32_t version = 0;
        VK_CHECK(vkEnumerateInstanceVersion(&version));
        spdlog::info("Loader supports Vulkan {}.{}.{}", VK_API_VERSION_MAJOR(version),
                     VK_API_VERSION_MINOR(version), VK_API_VERSION_PATCH(version));
        spdlog::info("Validation layers: {}", validation_enabled_ ? "on" : "off");
        spdlog::info("Instance extensions enabled ({}):", enabled_extensions_.size());
        for (const char* extension : enabled_extensions_) {
            spdlog::info("  {}", extension);
        }
        spdlog::info("Instance created. Nothing is drawn yet - a GPU comes next.");
    }

    vkc::Args args_;
    SDL_Window* window_ = nullptr;
    VkInstance instance_ = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debug_messenger_ = VK_NULL_HANDLE;
    std::vector<const char*> enabled_extensions_;
    bool validation_enabled_ = false;
    bool debug_utils_enabled_ = false;
};

}  // namespace

int main(int argc, char** argv) {
    try {
        InstanceApp app(vkc::parse_args(argc, argv));
        return app.run();
    } catch (const std::exception& error) {
        spdlog::error("{}", error.what());
        return EXIT_FAILURE;
    }
}
