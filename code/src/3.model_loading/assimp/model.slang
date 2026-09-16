// 3.1 Assimp: the same Phong shading as Part 2, cut down to the smallest thing that
// shows a shape clearly.
//
// There is nothing new in this file, and that is deliberate. The chapter's subject is
// where the vertices came from, so the shading is two directional lights and a flat
// albedo -- no maps, no light array, no specialisation constant. 3.2 puts the material
// back.

struct VSInput {
    [[vk::location(0)]] float3 position;
    [[vk::location(1)]] float3 normal;
};

struct VSOutput {
    float4 position : SV_Position;
    [[vk::location(0)]] float3 world_position;
    [[vk::location(1)]] float3 normal;
};

// A key light and a fill light, both directional, plus a flat ambient term. Two lights
// rather than one because a single light leaves half of a curved surface in the ambient
// floor, and a helmet is all curved surface.
//
// `ambient.w` carries the shininess. A float3 in a uniform block is padded to 16 bytes
// whether or not anything uses the fourth component (2.2), so this one is free.
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

// 2.6's push block exactly: 64 + 48 + 16 = 128 bytes.
struct PushConstants {
    float4x4 model;
    float3x3 normal_matrix;
    float4 colour;
};
[[vk::push_constant]] PushConstants push;

// 2.2's diffuse and specular terms, for parallel rays arriving along `to_light`.
float3 contribution(float3 to_light, float3 light_colour, float3 normal, float3 to_eye,
                    float3 albedo, float shininess) {
    float diffuse_amount = max(dot(normal, to_light), 0.0);
    float3 diffuse = light_colour * diffuse_amount * albedo;

    float3 reflected = reflect(-to_light, normal);
    float specular_amount = pow(max(dot(to_eye, reflected), 0.0), shininess);
    float3 specular = light_colour * specular_amount;

    return diffuse + specular;
}

[shader("vertex")]
VSOutput vertexMain(VSInput input) {
    VSOutput output;
    float4 world = mul(push.model, float4(input.position, 1.0));
    output.world_position = world.xyz;
    output.position = mul(globals.projection, mul(globals.view, world));
    output.normal = mul(push.normal_matrix, input.normal);
    return output;
}

[shader("fragment")]
float4 fragmentMain(VSOutput input) : SV_Target {
    // Normals arrive interpolated across the triangle, which is what makes a mesh of flat
    // faces read as a curved surface -- and the reason the model's own normals are worth
    // loading rather than deriving one per face.
    float3 normal = normalize(input.normal);
    float3 to_eye = normalize(globals.view_position.xyz - input.world_position);
    float3 albedo = push.colour.rgb;
    float shininess = globals.ambient.w;

    float3 colour = globals.ambient.rgb * albedo;
    colour += contribution(normalize(-globals.key_direction.xyz), globals.key_colour.rgb,
                           normal, to_eye, albedo, shininess);
    colour += contribution(normalize(-globals.fill_direction.xyz), globals.fill_colour.rgb,
                           normal, to_eye, albedo, shininess);

    return float4(colour, 1.0);
}
