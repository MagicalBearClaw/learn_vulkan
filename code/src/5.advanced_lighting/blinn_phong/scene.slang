// 5.1 The scene shader, lit by one point light, with the specular term measured two ways.
//
// Everything above shade() is 4.5's: a vertex read through a pointer, a DrawData picked
// out by index, a texture through the one descriptor left. What changed is the light --
// a lamp in the room rather than a sun -- and the specular term, which can be Phong's or
// Blinn's.

struct Vertex {
    float3 position;
    float3 normal;
    float2 uv;
};

// Matches Globals in main.cpp, 200 bytes, scalar layout.
struct Globals {
    float4x4 view;
    float4x4 projection;
    float4 view_position;
    float4 light_position;
    float4 light_colour;
    float4 ambient;
    float shininess;
    uint blinn;
};

struct DrawData {
    float4x4 model;
    float3x3 normal_matrix;
    float4 colour;
};

struct PushConstants {
    Globals* globals;
    DrawData* draws;
    Vertex* vertices;
    uint draw_index;
};
[[vk::push_constant]] PushConstants push;

[[vk::binding(0, 0)]] Sampler2D base_colour_map;

struct VSOutput {
    float4 position : SV_Position;
    [[vk::location(0)]] float3 world_position;
    [[vk::location(1)]] float3 normal;
    [[vk::location(2)]] float2 uv;
};

static const float kSpecularStrength = 0.5;

// 2.5's attenuation constants, unchanged. Chapter 5.7 replaces them with the
// inverse-square law.
static const float kConstantTerm = 1.0;
static const float kLinearTerm = 0.09;
static const float kQuadraticTerm = 0.032;

[shader("vertex")]
VSOutput vertexMain(uint vertex_id : SV_VulkanVertexID) {
    Vertex vertex = push.vertices[vertex_id];
    DrawData draw = push.draws[push.draw_index];

    VSOutput output;
    float4 world = mul(draw.model, float4(vertex.position, 1.0));
    output.world_position = world.xyz;
    output.position = mul(push.globals.projection, mul(push.globals.view, world));
    output.normal = mul(draw.normal_matrix, vertex.normal);
    output.uv = vertex.uv;
    return output;
}

// Phong's cosine: how closely the eye lines up with the light's mirror reflection.
float phong_cosine(float3 normal, float3 to_light, float3 to_eye) {
    float3 reflected = reflect(-to_light, normal);
    return dot(to_eye, reflected);
}

// Blinn's cosine: how closely the normal lines up with the direction halfway between the
// light and the eye -- the normal a mirror would need to send one into the other.
float blinn_cosine(float3 normal, float3 to_light, float3 to_eye) {
    float3 halfway = normalize(to_light + to_eye);
    return dot(normal, halfway);
}

float3 shade(VSOutput input, float3 albedo) {
    Globals globals = *push.globals;

    float3 normal = normalize(input.normal);
    float3 to_eye = normalize(globals.view_position.xyz - input.world_position);
    float3 offset = globals.light_position.xyz - input.world_position;
    float distance = length(offset);
    float3 to_light = offset / distance;

    float attenuation = 1.0 / (kConstantTerm + kLinearTerm * distance +
                               kQuadraticTerm * distance * distance);

    float facing = dot(normal, to_light);
    float diffuse_amount = max(facing, 0.0);

    float cosine = globals.blinn != 0 ? blinn_cosine(normal, to_light, to_eye)
                                      : phong_cosine(normal, to_light, to_eye);
    float specular_amount = pow(max(cosine, 0.0), globals.shininess);
    // A surface facing away from the light gets no highlight. Without this test Phong
    // leaks slivers at grazing angles and Blinn-Phong lights whole faces; the article
    // measures both.
    if (facing <= 0.0) {
        specular_amount = 0.0;
    }

    float3 diffuse = diffuse_amount * albedo;
    float3 specular = specular_amount * kSpecularStrength;
    float3 lit = globals.light_colour.rgb * (diffuse + specular) * attenuation;
    return globals.ambient.rgb * albedo + lit;
}

[shader("fragment")]
float4 fragmentMain(VSOutput input) : SV_Target {
    float3 tint = push.draws[push.draw_index].colour.rgb;
    float3 albedo = base_colour_map.Sample(input.uv).rgb * tint;
    return float4(shade(input, albedo), 1.0);
}

// The lamp itself: its colour, unlit.
[shader("fragment")]
float4 lampFragmentMain(VSOutput input) : SV_Target {
    return float4(push.draws[push.draw_index].colour.rgb, 1.0);
}
