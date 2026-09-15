// Lighting maps: the material's colours come from textures instead of constants.
//
// 2.3 gave every material one diffuse colour and one specular colour for its whole
// surface. A real object is rarely that uniform -- a crate is wood and steel, and they
// respond to light completely differently -- so the colours become per-texel lookups.
// The lighting arithmetic is untouched; only where two of its inputs come from changes.

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

struct Light {
    float4 position;
    float4 ambient;
    float4 diffuse;
    float4 specular;
};

// Set 0, per frame: unchanged from 2.3.
struct Globals {
    float4x4 view;
    float4x4 projection;
    float4 view_position;
    Light light;
};
[[vk::binding(0, 0)]] ConstantBuffer<Globals> globals;

// Set 1, per material: now three bindings of two different types.
//
// The diffuse map is the surface's colour, texel by texel. It is also used for the
// ambient term -- a surface's ambient colour is almost always its diffuse colour, darker,
// and the light's ambient component supplies the "darker".
[[vk::binding(0, 1)]] Sampler2D diffuse_map;

// The specular map says how much specular light each texel reflects. It is a greyscale
// image, and it is data rather than colour: main.cpp loads it as UNORM, not SRGB.
[[vk::binding(1, 1)]] Sampler2D specular_map;

// What is left of 2.3's material once its colours have become textures.
struct MaterialParams {
    float shininess;
};
[[vk::binding(2, 1)]] ConstantBuffer<MaterialParams> material;

struct PushConstants {
    float4x4 model;
    float3x3 normal_matrix;
};
[[vk::push_constant]] PushConstants push;

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

[shader("fragment")]
float4 fragmentMain(VSOutput input) : SV_Target {
    // Both samples first, before anything that could branch. Sample() needs screen-space
    // derivatives to pick a mip level, and those are only well defined when every
    // fragment in the 2x2 block takes the same path -- chapter 1.11's rule.
    float3 surface = diffuse_map.Sample(input.uv).rgb;
    float3 shine = specular_map.Sample(input.uv).rgb;

    float3 normal = normalize(input.normal);
    float3 to_light = normalize(globals.light.position.xyz - input.world_position);
    float3 to_eye = normalize(globals.view_position.xyz - input.world_position);

    float3 ambient = globals.light.ambient.rgb * surface;

    float diffuse_amount = max(dot(normal, to_light), 0.0);
    float3 diffuse = globals.light.diffuse.rgb * diffuse_amount * surface;

    float3 reflected = reflect(-to_light, normal);
    float specular_amount = pow(max(dot(to_eye, reflected), 0.0), material.shininess);
    float3 specular = globals.light.specular.rgb * specular_amount * shine;

    return float4(ambient + diffuse + specular, 1.0);
}

[shader("fragment")]
float4 lampFragmentMain(VSOutput input) : SV_Target {
    return float4(globals.light.specular.rgb, 1.0);
}
