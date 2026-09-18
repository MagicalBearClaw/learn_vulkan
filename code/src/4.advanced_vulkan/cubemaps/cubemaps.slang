// 4.4 The ordinary scene: 4.3's shader, with the glass taken out.
//
// Nothing in this file knows about cubemaps. It is here unchanged because the scene is
// drawn seven times a frame now -- once for the screen and once for each of the six
// faces of the live environment map -- and every one of those passes runs exactly this
// code. What differs between them is the view and projection matrices in `globals`,
// which is the entire point: a cubemap face is not a special kind of rendering, it is
// the same rendering aimed somewhere else.

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

// Set 0, per frame. One field longer than 4.3's: view_rotation is the view matrix with
// its translation removed, which the skybox needs and which is cheaper to build once on
// the CPU than to extract from a matrix in a shader.
struct Globals {
    float4x4 view;
    float4x4 projection;
    float4x4 view_rotation;
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
    float4 colour;
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
