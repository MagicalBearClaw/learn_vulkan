// One file, both stages.
//
// A Slang file holds every stage of a pipeline and compiles to a single SPIR-V module
// with one entry point per stage. The [shader("...")] attribute is what marks a
// function as an entry point and says which stage it is; the names vertexMain and
// fragmentMain are this project's convention, not a language rule, and main.cpp passes
// them to Vulkan as the entry point names.

// What the vertex stage hands to the fragment stage.
//
// SV_Position is a system-value semantic: it marks the one output every vertex shader
// must write, the clip-space position. Everything else is an ordinary varying, and
// [[vk::location(n)]] is the slot it travels in. Both stages use this one struct, so
// the two sides of the interface cannot drift apart -- in GLSL they are two separate
// lists of `out` and `in` declarations that you have to keep in step by hand.
struct VSOutput {
    float4 position : SV_Position;
    [[vk::location(0)]] float3 colour;
};

// The triangle's vertices live in the shader itself.
//
// There is no vertex buffer in this chapter, and that is deliberate: buffers, memory
// and staging are chapter 1.8's subject, and mixing them in here would mean learning
// two large things at once. The vertex index counts 0, 1, 2 across the three vertices
// that vkCmdDraw asked for, so it can just index a constant array.
//
// Clip space in Vulkan: x and y run -1..1 with y pointing DOWN the screen, and z runs
// 0..1. OpenGL has y pointing up and z running -1..1, which is the single most common
// thing to trip over when porting shaders. Here, -0.5 in y is the top of the screen.
static const float2 kPositions[3] = {
    float2( 0.0, -0.5),   // top
    float2( 0.5,  0.5),   // bottom right
    float2(-0.5,  0.5),   // bottom left
};

static const float3 kColours[3] = {
    float3(1.0, 0.0, 0.0),
    float3(0.0, 1.0, 0.0),
    float3(0.0, 0.0, 1.0),
};

// SV_VulkanVertexID, not SV_VertexID.
//
// The two differ, and the difference is worth knowing early. SV_VertexID is inherited
// from HLSL, where the index is relative to the draw: Slang implements it as
// `VertexIndex - BaseVertex`, which drags in SPIR-V's DrawParameters capability and so
// needs shaderDrawParameters enabled on the device. SV_VulkanVertexID is Vulkan's own
// VertexIndex builtin with nothing subtracted and nothing to enable. They agree
// whenever firstVertex is 0, which is why the mistake survives so long. SV_InstanceID
// and SV_VulkanInstanceID are the same story.
[shader("vertex")]
VSOutput vertexMain(uint vertex_index: SV_VulkanVertexID) {
    VSOutput output;
    output.position = float4(kPositions[vertex_index], 0.0, 1.0);
    output.colour = kColours[vertex_index];
    return output;
}

// One invocation per pixel covered by the triangle.
//
// input.colour is not the value any single vertex wrote. The hardware interpolates the
// three vertex outputs across the triangle's surface, weighted by how close the pixel
// is to each corner, which is why the result is a smooth gradient rather than three
// flat regions. That interpolation is free and it is the basis of almost everything in
// the lighting chapters.
//
// SV_Target is the semantic for a colour attachment output; SV_Target0 through
// SV_TargetN address the attachments individually once there is more than one.
[shader("fragment")]
float4 fragmentMain(VSOutput input) : SV_Target {
    return float4(input.colour, 1.0);
}
