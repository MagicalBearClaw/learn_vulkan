// 3.4 Bindless materials: the material stops being a binding and becomes an index.
//
// 3.3's fragment shader sampled `base_colour_map`, a single texture that the CPU rebound
// before every draw. This one samples `base_colour_maps[...]` -- an array holding every
// texture in the model at once, bound exactly once for the whole frame.
//
// The shader is the half of the change you can see. The half you cannot is that the draw
// loop on the CPU no longer calls vkCmdBindDescriptorSets at all.

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

// Set 0, per frame. Still 3.1's block, byte for byte, four chapters later.
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

// One material, as the GPU sees it.
//
// In 3.3 these two values arrived by different roads: the factor rode in the push
// constants and the texture was whatever happened to be bound. Here they sit together in
// a buffer, which is what lets a draw name a material with a single integer.
//
// `texture` is a separate index on purpose. Materials and textures are not in one-to-one
// correspondence -- FlightHelmet has two materials sharing one base colour image -- so
// collapsing the two indices into one would force a duplicate upload.
// The padding is three separate uints and not a uint3, which is not a style choice. In
// std430 a three-component vector aligns to 16 bytes, so a `uint3` here would start at
// offset 32 rather than 20 and make this struct 48 bytes against the CPU's 32. Every
// material after the first would then read from the wrong place -- which is exactly what
// happened the first time this chapter ran.
struct Material {
    float4 base_colour_factor;  // 0
    uint texture;               // 16
    uint pad0;                  // 20
    uint pad1;                  // 24
    uint pad2;                  // 28
};

// Set 1, the bindless set. Bound once, before the draw loop, and never touched again.
[[vk::binding(0, 1)]] StructuredBuffer<Material> materials;

// An unbounded array. Its real size is fixed when the descriptor set layout is created,
// which is after the model has been read and the textures counted; the shader does not
// need to know it and deliberately does not say.
//
// This declaration is what requires the runtimeDescriptorArray feature. Without it the
// validation layers reject the shader module rather than the draw, which is a much
// friendlier place to find out.
[[vk::binding(1, 1)]] Sampler2D base_colour_maps[];

// 116 bytes, down from 128. `colour`'s four floats at offset 112 are gone; the four-byte
// index that finds them in the buffer above takes their place.
struct PushConstants {
    float4x4 model;
    float3x3 normal_matrix;
    uint material;
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
    // Two indirections where 3.3 had none: the push constant names a material, and the
    // material names a texture.
    //
    // Both indices are the same for every fragment of every triangle in the draw -- they
    // come from a push constant, which cannot vary within a draw -- so this is a
    // *dynamically uniform* index and needs no NonUniformResourceIndex() around it. A
    // renderer that packed several materials into one draw would need one.
    Material material = materials[push.material];

    float3 albedo = base_colour_maps[material.texture].Sample(input.uv).rgb *
                    material.base_colour_factor.rgb;

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
