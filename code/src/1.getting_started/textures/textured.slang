struct VSInput {
    [[vk::location(0)]] float2 position;
    [[vk::location(1)]] float2 uv;
};

struct VSOutput {
    float4 position : SV_Position;
    [[vk::location(0)]] float2 uv;
};

struct PushConstants {
    float2 offset;
    float scale;
    float uv_scale;
};
[[vk::push_constant]] PushConstants push;

// Sampler2D is Slang's type for a combined image sampler: the view and the sampler
// arrive together in one descriptor. The binding and set pair with the
// VkDescriptorSetLayoutBinding in main.cpp exactly as a uniform buffer's would.
//
// Slang can also describe the two halves separately, as a Texture2D plus a
// SamplerState, which is what a VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE and a
// VK_DESCRIPTOR_TYPE_SAMPLER would be. This chapter uses the combined form because it
// is one descriptor to allocate and one object to reason about.
[[vk::binding(0, 0)]] Sampler2D tex;

[shader("vertex")]
VSOutput vertexMain(VSInput input) {
    VSOutput output;
    output.position = float4(input.position * push.scale + push.offset, 0.0, 1.0);

    // Multiplying the texture coordinate is how tiling works: uv_scale of 4 makes the
    // coordinate run 0..4 across the quad, and what happens beyond 1 is the sampler's
    // address mode -- REPEAT here, so it wraps four times.
    output.uv = input.uv * push.uv_scale;
    return output;
}

[shader("fragment")]
float4 fragmentMain(VSOutput input) : SV_Target {
    // Sample() does more than a memory read. The hardware compares this fragment's uv
    // with its neighbours' -- which is why fragments run in groups of at least 2x2 --
    // works out how fast the coordinate is changing across the screen, picks the mip
    // level where one texel is about one pixel, and blends between the two nearest
    // levels. All of that is one instruction.
    //
    // It also means Sample() inside non-uniform control flow is trouble: if some
    // fragments in the group took the other branch, their uv is undefined and so is the
    // derivative. Sample first, branch afterwards. (SampleLevel() names the mip level
    // explicitly and is safe anywhere, at the cost of choosing it yourself.)
    return tex.Sample(input.uv);
}
