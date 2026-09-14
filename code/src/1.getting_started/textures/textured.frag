#version 450

layout(location = 0) in vec2 frag_uv;

layout(location = 0) out vec4 out_colour;

// sampler2D is the GLSL type for a combined image sampler: the view and the sampler
// arrive together in one descriptor. `set` and `binding` pair with the
// VkDescriptorSetLayoutBinding in main.cpp exactly as the uniform buffer's did.
layout(set = 0, binding = 0) uniform sampler2D tex;

void main() {
    // texture() does more than a memory read. The hardware compares this fragment's uv
    // with its neighbours' -- which is why fragments run in groups of at least 2x2 --
    // works out how fast the coordinate is changing across the screen, picks the mip
    // level where one texel is about one pixel, and blends between the two nearest
    // levels. All of that is one instruction.
    //
    // It also means texture() inside non-uniform control flow is trouble: if some
    // fragments in the group took the other branch, their uv is undefined and so is the
    // derivative. Sample first, branch afterwards.
    out_colour = texture(tex, frag_uv);
}
