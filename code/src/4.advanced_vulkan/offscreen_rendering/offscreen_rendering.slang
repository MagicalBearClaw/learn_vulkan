// 4.3 The first pass: 4.2's scene, unchanged.
//
// This file is 4.2's shader with the cull-mode experiments taken out. Nothing in it
// knows or cares that its output is no longer going to the screen -- which is the point
// worth noticing. A fragment shader writes to SV_Target; what SV_Target *is* was decided
// outside the shader, in the VkRenderingInfo, and can be changed without touching a line
// of Slang.

struct VSInput {
    [[vk::location(0)]] float3 position;
    [[vk::location(1)]] float3 normal;
    [[vk::location(2)]] float2 uv;
};

struct VSOutput {
    float4 position : SV_Position;
    [[vk::location(0)]] float3 world_position;
    [[vk::location(1)]] float3 normal;
    [[vk::location(2)]] float2 uv;
};

// Set 0, per frame.
struct Globals {
    float4x4 view;
    float4x4 projection;
    float4 view_position;
    float4 sun_direction;
    float4 sun_colour;
    float4 ambient;  // rgb ambient, w shininess
};
[[vk::binding(0, 0)]] ConstantBuffer<Globals> globals;

// Set 1, per material.
[[vk::binding(0, 1)]] Sampler2D base_colour_map;

struct PushConstants {
    float4x4 model;
    float3x3 normal_matrix;
    float4 colour;  // the glass panes carry their transparency in .a
};
[[vk::push_constant]] PushConstants push;

static const float kSpecularStrength = 0.2;
static const float kCutoutThreshold = 0.5;

[shader("vertex")]
VSOutput vertexMain(VSInput input) {
    VSOutput output;
    float4 world = mul(push.model, float4(input.position, 1.0));
    output.world_position = world.xyz;
    output.position = mul(globals.projection, mul(globals.view, world));
    output.normal = mul(push.normal_matrix, input.normal);
    output.uv = input.uv;
    return output;
}

// 4.2's two-sided flip, for the flat sheets drawn with culling off.
float3 shade(VSOutput input, float3 albedo, bool two_sided) {
    float3 normal = normalize(input.normal);
    float3 to_eye = normalize(globals.view_position.xyz - input.world_position);
    if (two_sided && dot(normal, to_eye) < 0.0) {
        normal = -normal;
    }

    float3 to_sun = normalize(-globals.sun_direction.xyz);

    float diffuse_amount = max(dot(normal, to_sun), 0.0);
    float3 diffuse = globals.sun_colour.rgb * diffuse_amount * albedo;

    float3 reflected = reflect(-to_sun, normal);
    float specular_amount = pow(max(dot(to_eye, reflected), 0.0), globals.ambient.w);
    float3 specular = globals.sun_colour.rgb * specular_amount * kSpecularStrength;

    return globals.ambient.rgb * albedo + diffuse + specular;
}

[shader("fragment")]
float4 fragmentMain(VSOutput input) : SV_Target {
    float3 albedo = base_colour_map.Sample(input.uv).rgb * push.colour.rgb;
    return float4(shade(input, albedo, false), 1.0);
}

[shader("fragment")]
float4 cutoutFragmentMain(VSOutput input) : SV_Target {
    float4 sampled = base_colour_map.Sample(input.uv);
    if (sampled.a < kCutoutThreshold) {
        discard;
    }
    return float4(shade(input, sampled.rgb * push.colour.rgb, true), 1.0);
}

[shader("fragment")]
float4 flatFragmentMain(VSOutput input) : SV_Target {
    return push.colour;
}
