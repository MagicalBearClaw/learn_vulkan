cmake_minimum_required(VERSION 3.25)

project(learn_vulkan_com
    VERSION 0.1.0
    DESCRIPTION "Accompanying code for the LearnVulkan tutorial series"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "" FORCE)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(AddChapter)

# Every chapter binary lands in one place so tools/capture.py can find them by name.
set(LVK_BINARY_DIR "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${LVK_BINARY_DIR}")

# Assets live outside the build tree and are fetched by tools/bootstrap.py. Samples
# resolve them through this absolute path so they run from any working directory.
set(LVK_ASSET_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../assets" CACHE PATH "Tutorial asset directory")
cmake_path(NORMAL_PATH LVK_ASSET_DIR)

include(Dependencies)

add_subdirectory(common)
add_subdirectory(src)
