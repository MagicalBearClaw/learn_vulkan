// 5.5 The scene shader, with each surface's normal read from a texture.
//
// 5.4's shader. The vertex shader now carries a tangent through to the fragment shader,
// which builds a basis from it and the normal, reads a direction from the normal map in
// that basis, and lights with the result. The shadow pass is 5.4's, unchanged.

// 5.4's vertex, with a tangent: the direction of increasing u along the surface, and in
// w, which way the bitangent points.
struct Vertex {
    float3 position;
    float3 normal;
    float2 uv;
    float4 tangent;
};

static const uint kLampCount = 3;
static const uint kFaceCount = 6;

struct Lamp {
    float4 position;
    float4 colour;
};

// Matches Globals in main.cpp, 1424 bytes, scalar layout.
struct Globals {
    float4x4 view;
    float4x4 projection;
    float4 view_position;
    float4 ambient;
    Lamp lamps[kLampCount];
    float4x4 faces[kLampCount * kFaceCount];
    float shadow_near;
    float shadow_far;
    float shadow_texel;
    uint flags;
};

static const uint kNormalMapsOn = 1u << 0;
static const uint kFlipGreen = 1u << 1;
static const uint kShowNormals = 1u << 2;

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
    uint face_base;
};
[[vk::push_constant]] PushConstants push;

[[vk::binding(0, 0)]] Sampler2D base_colour_map;
// The surface's normals, one per texel, in the surface's own basis.
[[vk::binding(1, 0)]] Sampler2D normal_map;

[[vk::binding(0, 1)]] SamplerCubeShadow shadow_maps[kLampCount];

struct VSOutput {
    float4 position : SV_Position;
    [[vk::location(0)]] float3 world_position;
    [[vk::location(1)]] float3 normal;
    [[vk::location(2)]] float2 uv;
    [[vk::location(3)]] float4 tangent;
};

static const float kShininess = 64.0;
static const float kSpecularStrength = 0.5;

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
    // The tangent lies along the surface, so it moves with the model matrix, as an edge
    // of a triangle would. w = 0 because it is a direction: no translation. The normal
    // matrix is only for the normal, which is perpendicular to the surface.
    output.tangent = float4(mul(draw.model, float4(vertex.tangent.xyz, 0.0)).xyz,
                            vertex.tangent.w);
    return output;
}

// 5.4's shadow pass, unchanged.
[shader("vertex")]
float4 shadowVertexMain(uint vertex_id : SV_VulkanVertexID,
                        uint view : SV_ViewID) : SV_Position {
    Vertex vertex = push.vertices[vertex_id];
    DrawData draw = push.draws[push.draw_index];
    float4 world = mul(draw.model, float4(vertex.position, 1.0));
    return mul(push.globals.faces[push.face_base + view], world);
}

// 5.4's cube_depth(), unchanged.
float cube_depth(float3 offset) {
    float3 size = abs(offset);
    float along = max(size.x, max(size.y, size.z));
    float n = push.globals.shadow_near;
    float f = push.globals.shadow_far;
    return f / (f - n) - f * n / ((f - n) * along);
}

// 5.4's lamp_visibility(), with only its 3x3 filter: the chapter has no key for the
// others.
float lamp_visibility(uint lamp, float3 world_position) {
    Globals globals = *push.globals;

    float3 offset = world_position - globals.lamps[lamp].position.xyz;
    float depth = cube_depth(offset);
    if (depth > 1.0) {
        return 1.0;
    }

    float3 size = abs(offset);
    float along = max(size.x, max(size.y, size.z));
    float3 across_a;
    float3 across_b;
    if (along == size.x) {
        across_a = float3(0, 1, 0);
        across_b = float3(0, 0, 1);
    } else if (along == size.y) {
        across_a = float3(1, 0, 0);
        across_b = float3(0, 0, 1);
    } else {
        across_a = float3(1, 0, 0);
        across_b = float3(0, 1, 0);
    }
    float step = 2.0 * along * globals.shadow_texel;

    float sum = 0.0;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            float3 tap = offset + (float(x) * across_a + float(y) * across_b) * step;
            sum += shadow_maps[lamp].SampleCmpLevelZero(tap, depth);
        }
    }
    return sum / 9.0;
}

// The normal to light this point with: the interpolated vertex normal, or, with normal
// maps on, the normal map's direction carried out of the surface's basis into the
// world's.
float3 shading_normal(VSOutput input, float3 normal) {
    uint flags = push.globals.flags;
    if ((flags & kNormalMapsOn) == 0) {
        return normal;
    }

    // Interpolation leaves the tangent neither unit length nor quite perpendicular to
    // the interpolated normal. Removing its component along the normal, then
    // normalising, restores both: Gram-Schmidt, one step of it.
    float3 tangent = input.tangent.xyz;
    tangent = normalize(tangent - normal * dot(normal, tangent));
    // The third axis, up the image. w says whether up the image is this cross product or
    // its opposite; main.cpp worked out which, from the uvs.
    float3 bitangent = cross(normal, tangent) * input.tangent.w;

    // Each channel was stored as c * 0.5 + 0.5, to fit -1..1 into a UNORM's 0..1.
    float3 local = normal_map.Sample(input.uv).xyz * 2.0 - 1.0;
    if ((flags & kFlipGreen) != 0) {
        local.y = -local.y;
    }

    // The three columns of the basis, weighted by the three components: the matrix
    // whose columns are T, B and N, times the vector. Filtering and mipmapping average
    // texels, and an average of unit vectors is shorter than one, so it is normalised.
    return normalize(tangent * local.x + bitangent * local.y + normal * local.z);
}

// 5.4's lamp_light(), with two normals. The facing test asks whether the lamp is in
// front of the *surface*, and so uses the geometric normal: a normal map can tilt the
// shading normal towards a lamp that is behind the face.
float3 lamp_light(Lamp lamp, float3 world_position, float3 normal, float3 surface,
                  float3 to_eye, float3 albedo) {
    float3 offset = lamp.position.xyz - world_position;
    float distance = length(offset);
    float3 to_light = offset / distance;

    float attenuation = 1.0 / (kConstantTerm + kLinearTerm * distance +
                               kQuadraticTerm * distance * distance);

    if (dot(surface, to_light) <= 0.0) {
        return float3(0.0);
    }
    float facing = max(dot(normal, to_light), 0.0);

    float3 halfway = normalize(to_light + to_eye);
    float specular_amount = pow(max(dot(normal, halfway), 0.0), kShininess);

    float3 diffuse = facing * albedo;
    float3 specular = specular_amount * kSpecularStrength;
    return lamp.colour.rgb * (diffuse + specular) * attenuation;
}

float3 shade(VSOutput input, float3 surface, float3 normal, float3 albedo) {
    Globals globals = *push.globals;

    float3 to_eye = normalize(globals.view_position.xyz - input.world_position);

    float3 total = globals.ambient.rgb * albedo;
    for (uint i = 0; i < kLampCount; ++i) {
        total += lamp_light(globals.lamps[i], input.world_position, normal, surface,
                            to_eye, albedo) *
                 lamp_visibility(i, input.world_position);
    }
    return total;
}

[shader("fragment")]
float4 fragmentMain(VSOutput input) : SV_Target {
    float3 surface = normalize(input.normal);
    float3 normal = shading_normal(input, surface);

    // A direction drawn as a colour: each component from -1..1 to 0..1. Facing +x is
    // red, +y green, +z blue.
    if ((push.globals.flags & kShowNormals) != 0) {
        return float4(normal * 0.5 + 0.5, 1.0);
    }

    float3 tint = push.draws[push.draw_index].colour.rgb;
    float3 albedo = base_colour_map.Sample(input.uv).rgb * tint;
    return float4(shade(input, surface, normal, albedo), 1.0);
}

[shader("fragment")]
float4 lampFragmentMain(VSOutput input) : SV_Target {
    return float4(push.draws[push.draw_index].colour.rgb, 1.0);
}
