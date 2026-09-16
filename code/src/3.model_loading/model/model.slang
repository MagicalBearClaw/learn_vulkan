// 3.3 Model: 3.2's shading, unchanged.
//
// This chapter's entire subject is on the CPU -- where each mesh sits, and how the file
// says so. The shader is the control: if the picture changes and this file did not, the
// change came from the matrix in the push block.
//
// One line of it does move, though. `push.model` was the same matrix for every draw in
// 3.2, because every mesh sat at the origin once Assimp had baked the transforms in. It
// is now the model matrix times *that node's* accumulated transform, and the shader
// cannot tell the difference -- which is the point.

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

// Set 0, per frame. Byte for byte 3.1's block, still.
struct Globals {
    float4x4 view;
    float4x4 projection;
    float4 view_position;

    float4 key_direction;
    float4 key_colour;
    float4 fill_direction;
    float4 fill_colour;

    float4 ambient;
};
[[vk::binding(0, 0)]] ConstantBuffer<Globals> globals;

// Set 1, per material, rebound inside the draw loop exactly as in 3.2.
[[vk::binding(0, 1)]] Sampler2D base_colour_map;

// 3.2's push block at 128 bytes. What changed is not its shape but what the CPU puts in
// it: `model` is now per draw rather than per frame, and `normal_matrix` with it.
struct PushConstants {
    float4x4 model;
    float3x3 normal_matrix;
    float4 colour;
};
[[vk::push_constant]] PushConstants push;

// 3.2's compromise, unchanged: glTF describes roughness and metalness, not a Phong
// exponent, so one number stands in for the whole surface until Part 6.
static const float kSpecularStrength = 0.25;

float3 contribution(float3 to_light, float3 light_colour, float3 normal, float3 to_eye,
                    float3 albedo, float shininess) {
    float diffuse_amount = max(dot(normal, to_light), 0.0);
    float3 diffuse = light_colour * diffuse_amount * albedo;

    float3 reflected = reflect(-to_light, normal);
    float specular_amount = pow(max(dot(to_eye, reflected), 0.0), shininess);
    float3 specular = light_colour * specular_amount * kSpecularStrength;

    return diffuse + specular;
}

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
    float3 albedo = base_colour_map.Sample(input.uv).rgb * push.colour.rgb;

    float3 normal = normalize(input.normal);
    float3 to_eye = normalize(globals.view_position.xyz - input.world_position);
    float shininess = globals.ambient.w;

    float3 colour = globals.ambient.rgb * albedo;
    colour += contribution(normalize(-globals.key_direction.xyz), globals.key_colour.rgb,
                           normal, to_eye, albedo, shininess);
    colour += contribution(normalize(-globals.fill_direction.xyz), globals.fill_colour.rgb,
                           normal, to_eye, albedo, shininess);

    return float4(colour, 1.0);
}
