// 5.4 The scene shader, lit by three lamps, each with a cube of depth to say where it
// cannot reach.
//
// vertexMain is 5.2's. shadowVertexMain is 5.3's, choosing among eighteen matrices
// instead of using one. The lighting is 5.2's sum over lamps, with each lamp's share
// multiplied by how much of that lamp the point can see.

struct Vertex {
    float3 position;
    float3 normal;
    float2 uv;
};

static const uint kLampCount = 3;
static const uint kFaceCount = 6;

struct Lamp {
    float4 position;
    float4 colour;
};

// Matches Globals in main.cpp, 1428 bytes, scalar layout.
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
    uint filter;
};

static const uint kShadowsOn = 1u << 0;

static const uint kFilterOneTexel = 0;
static const uint kFilterBilinear = 1;
static const uint kFilterPcf3x3 = 2;

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

// One cube per lamp, each through a comparison sampler. A cube is read by direction, not
// by a uv pair: the direction picks the face and the texel on it.
[[vk::binding(0, 1)]] SamplerCubeShadow shadow_maps[kLampCount];

struct VSOutput {
    float4 position : SV_Position;
    [[vk::location(0)]] float3 world_position;
    [[vk::location(1)]] float3 normal;
    [[vk::location(2)]] float2 uv;
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
    return output;
}

// The shadow pass, for both ways of filling a cube. SV_ViewID is which view of a
// multiview pass this run of the shader belongs to: 0 to 5 when the pass draws all six
// faces at once, and always 0 in an ordinary pass, where face_base alone names the face.
[shader("vertex")]
float4 shadowVertexMain(uint vertex_id : SV_VulkanVertexID,
                        uint view : SV_ViewID) : SV_Position {
    Vertex vertex = push.vertices[vertex_id];
    DrawData draw = push.draws[push.draw_index];
    float4 world = mul(draw.model, float4(vertex.position, 1.0));
    return mul(push.globals.faces[push.face_base + view], world);
}

// The depth a cube face stored for a point at `offset` from its lamp. The face is the
// one whose axis is offset's largest component, and that component is the point's
// distance along the face camera's line of sight; the rest is that camera's projection.
float cube_depth(float3 offset) {
    float3 size = abs(offset);
    float along = max(size.x, max(size.y, size.z));
    float n = push.globals.shadow_near;
    float f = push.globals.shadow_far;
    return f / (f - n) - f * n / ((f - n) * along);
}

// How much of lamp `lamp` reaches this point: 1 in full light, 0 in full shadow, and
// fractions along the edges when the filter reads more than one texel.
float lamp_visibility(uint lamp, float3 world_position) {
    Globals globals = *push.globals;
    if ((globals.flags & kShadowsOn) == 0) {
        return 1.0;
    }

    float3 offset = world_position - globals.lamps[lamp].position.xyz;
    float depth = cube_depth(offset);

    // Beyond the far plane the comparison would always fail. The far plane reaches the
    // corners of the floor, so this only matters to a reader who moves a lamp outside.
    if (depth > 1.0) {
        return 1.0;
    }

    if (globals.filter != kFilterPcf3x3) {
        return shadow_maps[lamp].SampleCmpLevelZero(offset, depth);
    }

    // 5.3's nine bilinear reads, one texel apart, now on a cube face. The two directions
    // across the face are the two axes that are not its own, and one texel along them
    // is 2 / size of the face at distance 1 -- so 2 * along / size at this distance.
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
            // Every tap is compared against this point's own depth, as in 5.3. A tap
            // that crosses an edge reads the next face, whose depths are measured along
            // another axis -- but near an edge the two distances are nearly equal,
            // which is what being near the edge means.
            float3 tap = offset + (float(x) * across_a + float(y) * across_b) * step;
            sum += shadow_maps[lamp].SampleCmpLevelZero(tap, depth);
        }
    }
    return sum / 9.0;
}

// 5.2's lamp_light(), unchanged: one lamp, without the ambient.
float3 lamp_light(Lamp lamp, float3 world_position, float3 normal, float3 to_eye,
                  float3 albedo) {
    float3 offset = lamp.position.xyz - world_position;
    float distance = length(offset);
    float3 to_light = offset / distance;

    float attenuation = 1.0 / (kConstantTerm + kLinearTerm * distance +
                               kQuadraticTerm * distance * distance);

    float facing = dot(normal, to_light);
    if (facing <= 0.0) {
        return float3(0.0);
    }

    float3 halfway = normalize(to_light + to_eye);
    float specular_amount = pow(max(dot(normal, halfway), 0.0), kShininess);

    float3 diffuse = facing * albedo;
    float3 specular = specular_amount * kSpecularStrength;
    return lamp.colour.rgb * (diffuse + specular) * attenuation;
}

float3 shade(VSOutput input, float3 albedo) {
    Globals globals = *push.globals;

    float3 normal = normalize(input.normal);
    float3 to_eye = normalize(globals.view_position.xyz - input.world_position);

    float3 total = globals.ambient.rgb * albedo;
    // Unrolled, so that every index into shadow_maps is a constant. An array of
    // descriptors indexed by a value computed at run time asks the device for a feature
    // this series has not enabled; a constant asks for nothing.
    [ForceUnroll]
    for (uint i = 0; i < kLampCount; ++i) {
        total += lamp_light(globals.lamps[i], input.world_position, normal, to_eye,
                            albedo) *
                 lamp_visibility(i, input.world_position);
    }
    return total;
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
