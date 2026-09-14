#version 450

// Specialisation constants.
//
// These look like ordinary `const` declarations with a default value, and if the host
// says nothing that is exactly what they are. But main.cpp supplies values for ids 0
// and 1 when it creates the pipeline, and those values replace the defaults before the
// driver compiles the SPIR-V to machine code.
//
// The result is a genuine compile-time constant: the branch below costs nothing at run
// time, because only one side of it survives compilation. Changing the value means
// rebuilding the pipeline, which is exactly the trade -- free in the shader, expensive
// to change. For values that change per frame you want push constants or a uniform
// buffer instead, and those are chapter 1.10.
layout(constant_id = 0) const int kShadingMode = 0;    // 0 = smooth, 1 = flat
layout(constant_id = 1) const float kCheckerSize = 48.0;

layout(location = 0) in vec3 frag_colour;
layout(location = 1) flat in vec3 flat_colour;

layout(location = 0) out vec4 out_colour;

void main() {
    vec3 base = (kShadingMode == 1) ? flat_colour : frag_colour;

    // gl_FragCoord is a built-in the hardware fills in: xy is this fragment's position
    // in pixels, with the origin at the top-left corner of the framebuffer, matching
    // Vulkan's downward y. z is the depth that will be written.
    //
    // Because it is measured in pixels and not in the quad's own coordinates, the
    // checkerboard is fixed to the screen -- resize the window and the quad changes
    // size while the squares do not. Chapter 1.11 introduces texture coordinates, which
    // are the way to attach a pattern to the surface instead.
    vec2 cell = floor(gl_FragCoord.xy / kCheckerSize);
    float checker = mod(cell.x + cell.y, 2.0);

    // mix(a, b, t) is linear interpolation, and it is everywhere in shader code.
    // Here it darkens every other square to 55% brightness.
    out_colour = vec4(mix(base, base * 0.55, checker), 1.0);
}
