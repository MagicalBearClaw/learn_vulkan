// 4.1 Depth and stencil: three fragment shaders, and almost no shading.
//
// This chapter's subject is not in here. Every visible effect it demonstrates -- what
// survives the depth test, which fragments the stencil test lets through, how far a
// polygon is pushed before it is compared -- is decided by pipeline state on the CPU
// side, after the fragment shader has already run.
//
// That is worth saying out loud, because it is the opposite of how the lighting
// chapters worked. There, changing the picture meant changing the shader. Here the
// shader is a constant and the picture changes anyway.

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

// Set 0, per frame. One sun and an ambient term: this chapter has no lighting to teach,
// and a scene needs just enough shading for the shapes to read.
struct Globals {
    float4x4 view;
    float4x4 projection;
    float4 view_position;
    float4 sun_direction;
    float4 sun_colour;
    float4 ambient;  // rgb ambient, w shininess
};
[[vk::binding(0, 0)]] ConstantBuffer<Globals> globals;

// Set 1, per material. One texture, because one is all that changes between the crate
// and the floor.
[[vk::binding(0, 1)]] Sampler2D base_colour_map;

struct PushConstants {
    float4x4 model;
    float3x3 normal_matrix;
    float4 colour;
};
[[vk::push_constant]] PushConstants push;

static const float kSpecularStrength = 0.2;

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

// The textured, lit surfaces: the floor and the two crates. `push.colour` tints it,
// and is white everywhere in this chapter.
[shader("fragment")]
float4 fragmentMain(VSOutput input) : SV_Target {
    float3 albedo = base_colour_map.Sample(input.uv).rgb * push.colour.rgb;

    float3 normal = normalize(input.normal);
    float3 to_eye = normalize(globals.view_position.xyz - input.world_position);
    float3 to_sun = normalize(-globals.sun_direction.xyz);

    float diffuse_amount = max(dot(normal, to_sun), 0.0);
    float3 diffuse = globals.sun_colour.rgb * diffuse_amount * albedo;

    float3 reflected = reflect(-to_sun, normal);
    float specular_amount = pow(max(dot(to_eye, reflected), 0.0), globals.ambient.w);
    float3 specular = globals.sun_colour.rgb * specular_amount * kSpecularStrength;

    return float4(globals.ambient.rgb * albedo + diffuse + specular, 1.0);
}

// The decal and the outline. Both are flat colour, and neither is lit: a decal painted
// on the floor takes the floor's light, and an outline is not a surface at all.
//
// One entry point serves both because what distinguishes them is entirely pipeline
// state -- the decal is depth-biased and stencil-blind, the outline is depth-blind and
// stencil-tested -- which is the chapter's whole point in one shader function.
[shader("fragment")]
float4 flatFragmentMain(VSOutput input) : SV_Target {
    return push.colour;
}
