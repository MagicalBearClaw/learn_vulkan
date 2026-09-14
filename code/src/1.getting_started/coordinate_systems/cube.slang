struct VSInput {
    [[vk::location(0)]] float3 position;
    [[vk::location(1)]] float2 uv;
};

struct VSOutput {
    float4 position : SV_Position;
    [[vk::location(0)]] float2 uv;
};

struct Globals {
    float4x4 view;
    float4x4 projection;
};
[[vk::binding(0, 0)]] ConstantBuffer<Globals> globals;

struct PushConstants {
    float4x4 model;
};
[[vk::push_constant]] PushConstants push;

[[vk::binding(1, 0)]] Sampler2D tex;

[shader("vertex")]
VSOutput vertexMain(VSInput input) {
    VSOutput output;

    // The full chain. Each mul() applies one matrix to the result of the last, so the
    // innermost happens first and the transforms read inside out:
    //
    //   model space  --model-->  world space
    //                --view-->   view space   (camera at the origin, looking down -z)
    //                --proj-->   clip space   (what SV_Position holds)
    //
    // Written with GLSL's `*` this is the familiar right-to-left
    // `projection * view * model * position`; nesting mul() calls is the same order
    // spelled differently.
    //
    // The hardware then divides by w to get normalised device coordinates and maps
    // those onto the viewport. That divide is what makes distant things small: the
    // projection matrix puts the distance to the camera into w, so a vertex twice as
    // far away is divided by twice as much.
    //
    // Multiplying the three matrices together on the CPU, once per object, would save
    // the GPU two matrix multiplies per vertex. Worth doing when there are many
    // vertices; kept separate here because the three stages are the lesson.
    float4 world = mul(push.model, float4(input.position, 1.0));
    float4 view_space = mul(globals.view, world);
    output.position = mul(globals.projection, view_space);

    output.uv = input.uv;
    return output;
}

[shader("fragment")]
float4 fragmentMain(VSOutput input) : SV_Target {
    return tex.Sample(input.uv);
}
