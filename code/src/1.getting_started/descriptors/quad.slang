struct VSInput {
    [[vk::location(0)]] float2 position;
    [[vk::location(1)]] float3 colour;
};

struct VSOutput {
    float4 position : SV_Position;
    [[vk::location(0)]] float3 colour;
};

// A uniform buffer, reached through descriptor set 0, binding 0.
//
// ConstantBuffer<T> is Slang's name for what Vulkan calls a uniform buffer: a small,
// read-only block the shader reads through a descriptor. The two numbers in
// [[vk::binding(binding, set)]] pair with the VkDescriptorSetLayoutBinding in main.cpp
// and with the set index passed to vkCmdBindDescriptorSets. The struct's name is not
// matched against anything on the host; only the numbers are.
//
// Declared once, at file scope, and both entry points below read it. In GLSL each stage
// repeats the block declaration in its own file and the two copies must agree.
struct Globals {
    float4 ambient;
    float time;
};
[[vk::binding(0, 0)]] ConstantBuffer<Globals> globals;

// The push constant block. There is exactly one per pipeline layout -- not one per
// binding -- so everything a draw needs goes in this single struct.
struct PushConstants {
    float2 offset;
    float scale;
    float phase;
};
[[vk::push_constant]] PushConstants push;

[shader("vertex")]
VSOutput vertexMain(VSInput input) {
    // A gentle pulse so the two data paths are both visibly live: `time` is shared by
    // all four quads and comes from the uniform buffer, `phase` differs per quad and
    // comes from a push constant.
    float pulse = 0.85 + 0.15 * sin(globals.time + push.phase);

    VSOutput output;
    output.position = float4(input.position * push.scale * pulse + push.offset, 0.0, 1.0);
    output.colour = input.colour;
    return output;
}

[shader("fragment")]
float4 fragmentMain(VSOutput input) : SV_Target {
    float brightness = 0.6 + 0.4 * sin(globals.time + push.phase);

    // The ambient term comes from the uniform buffer and is identical for all four
    // quads; the brightness differs because each quad pushed its own phase.
    return float4(input.colour * brightness + globals.ambient.rgb, 1.0);
}
