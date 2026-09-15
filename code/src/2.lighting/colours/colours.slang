// Two pipelines, one file.
//
// The lit cube and the lamp cube are the same geometry drawn two ways, so they share a
// vertex stage and differ only in the fragment stage. This file declares three entry
// points; main.cpp builds two pipelines from the one SPIR-V module, picking the
// fragment entry point by name.
//
// That is the payoff of the pName field chapter 1.7 introduced. In GLSL this would be
// three files and two of them nearly identical.

struct VSInput {
    [[vk::location(0)]] float3 position;
};

struct VSOutput {
    float4 position : SV_Position;
};

// Per-frame data: the same for every draw in the frame.
//
// This is set 0, and from here on it stays set 0 and keeps that meaning. Chapter 2.3
// adds set 1 for per-material data, and the split is the point: the camera changes once
// per frame, a material changes once per object, and rebinding the cheap one should not
// mean rewriting the expensive one.
struct Globals {
    float4x4 view;
    float4x4 projection;

    // The colour of the light. White here, and changing it is the first exercise.
    float4 light_colour;
};
[[vk::binding(0, 0)]] ConstantBuffer<Globals> globals;

// Per-object data, small enough to go in push constants rather than a buffer.
struct PushConstants {
    float4x4 model;
    float4 object_colour;
};
[[vk::push_constant]] PushConstants push;

// Shared by both pipelines. Neither the lit cube nor the lamp needs anything from the
// vertex stage beyond a clip-space position -- that changes in 2.2, when the fragment
// stage starts needing normals and a world position.
[shader("vertex")]
VSOutput vertexMain(VSInput input) {
    VSOutput output;
    float4 world = mul(push.model, float4(input.position, 1.0));
    output.position = mul(globals.projection, mul(globals.view, world));
    return output;
}

// The lit object.
//
// This is the whole of this chapter's "lighting": multiply the object's colour by the
// light's colour, component by component. No normals, no direction, no distance -- every
// face comes out the same flat shade, which is exactly the point. It is the wrong
// answer, and seeing precisely how it is wrong is what makes 2.2's diffuse term
// obviously right.
//
// Why multiply? Because an object's colour *is* the fraction of each wavelength it
// reflects. Coral under white light reflects most red, some green, little blue. Under a
// light with no blue in it, the blue it fails to reflect was never there to begin with,
// and the same multiply gives you that for free.
[shader("fragment")]
float4 fragmentMain(VSOutput input) : SV_Target {
    float3 result = push.object_colour.rgb * globals.light_colour.rgb;
    return float4(result, 1.0);
}

// The lamp.
//
// A light source is not lit by anything -- it is the thing doing the lighting -- so it
// ignores the shading model entirely and draws at the light's own colour. It exists to
// make the light's position visible; without it you are guessing where the illumination
// comes from.
[shader("fragment")]
float4 lampFragmentMain(VSOutput input) : SV_Target {
    return float4(globals.light_colour.rgb, 1.0);
}
