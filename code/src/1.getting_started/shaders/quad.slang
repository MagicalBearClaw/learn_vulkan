// Inputs from the vertex buffer. The [[vk::location(n)]] numbers pair with the
// VkVertexInputAttributeDescription entries in main.cpp.
struct VSInput {
    [[vk::location(0)]] float2 position;
    [[vk::location(1)]] float3 colour;
};

// The vertex-to-fragment interface.
//
// These location numbers are a completely separate numbering from the input ones
// above: they are slots between two stages, and nothing else. Because both entry points
// below take or return this one struct, the two ends are the same declaration -- there
// is no second list to keep in step.
struct VSOutput {
    // SV_Position is the clip-space position on the way out of the vertex stage. Read
    // as a fragment input it becomes something different: the fragment's position in
    // pixels, with the origin at the top-left corner of the framebuffer, matching
    // Vulkan's downward y. z is the depth that will be written. (GLSL spells these
    // gl_Position and gl_FragCoord and keeps them separate.)
    float4 position : SV_Position;

    [[vk::location(0)]] float3 colour;

    // The same value, sent again without interpolation.
    //
    // `nointerpolation` means every fragment of a triangle receives the value from one
    // nominated vertex -- the provoking vertex, which in Vulkan is the first of the
    // triangle -- rather than a weighted blend of all three. The fragment shader chooses
    // between the two, so you can see the difference without changing the geometry.
    [[vk::location(1)]] nointerpolation float3 flat_colour;
};

// Specialisation constants.
//
// These look like ordinary constants with a default value, and if the host says nothing
// that is exactly what they are. But main.cpp supplies values for ids 0 and 1 when it
// creates the pipeline, and those values replace the defaults before the driver
// compiles the SPIR-V to machine code.
//
// The result is a genuine compile-time constant: the branch below costs nothing at run
// time, because only one side of it survives compilation. Changing the value means
// rebuilding the pipeline, which is exactly the trade -- free in the shader, expensive
// to change. For values that change per frame you want push constants or a uniform
// buffer instead, and those are chapter 1.10.
[[vk::constant_id(0)]] const int kShadingMode = 0;      // 0 = smooth, 1 = flat
[[vk::constant_id(1)]] const float kCheckerSize = 48.0;

[shader("vertex")]
VSOutput vertexMain(VSInput input) {
    VSOutput output;

    // SV_Position is the one output every vertex shader must write. It is in clip
    // space: the hardware divides by w to get normalised device coordinates, then maps
    // those onto the viewport.
    //
    // w = 1.0 makes that division a no-op, which is right for 2D. Once there is a
    // perspective projection (chapter 1.13) w carries the depth and the division is
    // what makes distant things small.
    output.position = float4(input.position, 0.0, 1.0);

    output.colour = input.colour;
    output.flat_colour = input.colour;
    return output;
}

[shader("fragment")]
float4 fragmentMain(VSOutput input) : SV_Target {
    float3 base = (kShadingMode == 1) ? input.flat_colour : input.colour;

    // Because the checker is measured in pixels and not in the quad's own coordinates,
    // it is fixed to the screen -- resize the window and the quad changes size while the
    // squares do not. Chapter 1.11 introduces texture coordinates, which are the way to
    // attach a pattern to the surface instead.
    float2 cell = floor(input.position.xy / kCheckerSize);
    float checker = fmod(cell.x + cell.y, 2.0);

    // lerp(a, b, t) is linear interpolation, and it is everywhere in shader code. Here
    // it darkens every other square to 55% brightness. (GLSL spells it mix.)
    return float4(lerp(base, base * 0.55, checker), 1.0);
}
