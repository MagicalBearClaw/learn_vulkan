// 3.2 Mesh: 3.1's shading with the albedo taken from the material's base colour map.
//
// The lighting arithmetic is 3.1's, unchanged. What changed is one input: `albedo` was a
// push constant the sample chose, and is now a texture the file chose, multiplied by the
// material's own base colour factor.
//
// This is 2.4's lesson arriving by a different road. There the diffuse map was a PNG
// this project drew; here it is a JPEG that was already inside the model, and the shader
// cannot tell the difference.

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

// Set 0, per frame. Byte for byte 3.1's block: a key light, a fill light, an ambient
// term, and the shininess riding in `ambient.w`.
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

// Set 1, per material -- and "per material" now means the set is rebound inside the
// draw loop rather than once for the frame, because a model has several.
//
// glTF calls this the base colour texture and 2.4 called it the diffuse map. They are
// the same thing for an untextured-lighting model like this one: the surface's own
// colour, texel by texel, before any light touches it.
[[vk::binding(0, 1)]] Sampler2D base_colour_map;

// 3.1's push block, unchanged at 128 bytes. `colour` has changed meaning: it was the
// flat albedo the sample picked, and is now the material's base colour factor, which
// glTF defines as a multiplier on the texture.
struct PushConstants {
    float4x4 model;
    float3x3 normal_matrix;
    float4 colour;
};
[[vk::push_constant]] PushConstants push;

// How much of a light each texel reflects as a highlight.
//
// 2.4 read this from a specular map, and this model has none: glTF describes a surface
// with roughness and metalness instead, which is Part 6's subject. Until then one number
// stands in for the whole surface.
//
// It is not the arbitrary constant it looks like. 3.1's shading was written against a
// flat mid-grey albedo, where a full-strength highlight reads as shine; against this
// model's real base colour -- which is dark, worn and mostly metal -- the same highlight
// swamps the texture and the helmet turns to white plastic. The texture is what the
// chapter is for, so the highlight gets out of its way.
static const float kSpecularStrength = 0.25;

// 3.1's contribution(), with that one factor added.
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
    // The one line the vertex stage gains over 3.1. The UVs were in the vertex buffer
    // all along; nothing was reading them.
    output.uv = input.uv;
    return output;
}

[shader("fragment")]
float4 fragmentMain(VSOutput input) : SV_Target {
    // Sampled before any branch, as 1.11 explained: Sample() picks its mip level from
    // screen-space derivatives, and those are only well defined when every fragment in
    // the 2x2 quad reaches this line.
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
