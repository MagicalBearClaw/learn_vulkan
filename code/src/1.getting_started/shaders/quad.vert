#version 450

// The version directive is required and it is always a GLSL version, never a Vulkan
// one. 450 means GLSL 4.50, which is the dialect Vulkan's GLSL is based on.

// Inputs from the vertex buffer. `location` pairs with the
// VkVertexInputAttributeDescription entries in main.cpp.
layout(location = 0) in vec2 in_position;
layout(location = 1) in vec3 in_colour;

// Outputs to the fragment shader. These `location` numbers are a completely separate
// numbering from the input ones above: they pair with the fragment shader's `in`
// declarations, and nothing else.
layout(location = 0) out vec3 frag_colour;

// The same value, sent again without interpolation.
//
// `flat` means every fragment of a triangle receives the value from one nominated
// vertex -- the provoking vertex, which in Vulkan is the first of the triangle -- rather
// than a weighted blend of all three. The fragment shader chooses between the two, so
// you can see the difference without changing the geometry.
layout(location = 1) flat out vec3 flat_colour;

void main() {
    // gl_Position is the one output every vertex shader must write. It is in clip
    // space: the hardware divides by w to get normalised device coordinates, then maps
    // those onto the viewport.
    //
    // w = 1.0 makes that division a no-op, which is right for 2D. Once there is a
    // perspective projection (chapter 1.13) w carries the depth and the division is
    // what makes distant things small.
    gl_Position = vec4(in_position, 0.0, 1.0);

    frag_colour = in_colour;
    flat_colour = in_colour;
}
