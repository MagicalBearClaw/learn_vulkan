// 5.3 The scene shader, lit by the sun, with the sun's view of the scene to say where it
// cannot reach.
//
// Three vertex shaders share this file now. vertexMain is 5.2's. shadowVertexMain is the
// whole of the shadow pass: it places each vertex where the sun sees it and stops, with
// no fragment shader behind it. insetVertexMain draws key 4's picture of the shadow map.

struct Vertex {
    float3 position;
    float3 normal;
    float2 uv;
};

// Matches Globals in main.cpp, 268 bytes, scalar layout.
struct Globals {
    float4x4 view;
    float4x4 projection;
    float4x4 light_view_projection;
    float4 view_position;
    float4 ambient;
    float4 sun_direction;
    float4 sun_colour;
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
};
[[vk::push_constant]] PushConstants push;

[[vk::binding(0, 0)]] Sampler2D base_colour_map;

// The shadow map through a comparison sampler: reading it returns not a depth but the
// fraction of the texels read whose depth passed the sampler's comparison.
[[vk::binding(0, 1)]] Sampler2DShadow shadow_map;
// The same image through a plain sampler, which returns the stored depth. Key 4 only.
[[vk::binding(1, 1)]] Sampler2D shadow_depth;

struct VSOutput {
    float4 position : SV_Position;
    [[vk::location(0)]] float3 world_position;
    [[vk::location(1)]] float3 normal;
    [[vk::location(2)]] float2 uv;
};

static const float kShininess = 64.0;
static const float kSpecularStrength = 0.5;

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

// The shadow pass. The same vertex, the same model matrix, and the sun's matrix in place
// of the camera's. The rasteriser's depth is all this pass keeps, so there is nothing
// else to output.
[shader("vertex")]
float4 shadowVertexMain(uint vertex_id : SV_VulkanVertexID) : SV_Position {
    Vertex vertex = push.vertices[vertex_id];
    DrawData draw = push.draws[push.draw_index];
    float4 world = mul(draw.model, float4(vertex.position, 1.0));
    return mul(push.globals.light_view_projection, world);
}

// How much of the sun reaches this point: 1 in full sun, 0 in full shadow, and fractions
// along the edges when the filter reads more than one texel.
float sunlight(float3 world_position) {
    Globals globals = *push.globals;
    if ((globals.flags & kShadowsOn) == 0) {
        return 1.0;
    }

    // The point as the shadow pass saw it: clip space, then -- after the divide, which is
    // by 1 for an orthographic projection but costs nothing to keep -- normalised device
    // coordinates. x and y are -1..1 across the map; z is already 0..1, the value the
    // depth attachment stored.
    float4 clip = mul(globals.light_view_projection, float4(world_position, 1.0));
    float3 ndc = clip.xyz / clip.w;
    float2 uv = ndc.xy * 0.5 + 0.5;
    float depth = ndc.z;

    // Beyond the far plane the comparison would always fail. Nothing in this scene is
    // there, and the article explains why that is a property of the box, not luck.
    if (depth > 1.0) {
        return 1.0;
    }

    if (globals.filter != kFilterPcf3x3) {
        // One read. Through the nearest sampler that is one texel, 0 or 1; through the
        // linear sampler it is four, compared one by one and then blended.
        return shadow_map.SampleCmpLevelZero(uv, depth);
    }

    // Nine bilinear reads, one texel apart: a 4x4 block of texels, each compared.
    float sum = 0.0;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            float2 offset = float2(x, y) * globals.shadow_texel;
            sum += shadow_map.SampleCmpLevelZero(uv + offset, depth);
        }
    }
    return sum / 9.0;
}

float3 shade(VSOutput input, float3 albedo) {
    Globals globals = *push.globals;

    float3 normal = normalize(input.normal);
    float3 to_eye = normalize(globals.view_position.xyz - input.world_position);
    float3 to_sun = globals.sun_direction.xyz;

    float3 ambient = globals.ambient.rgb * albedo;

    float facing = dot(normal, to_sun);
    if (facing <= 0.0) {
        return ambient;
    }

    float3 halfway = normalize(to_sun + to_eye);
    float specular_amount = pow(max(dot(normal, halfway), 0.0), kShininess);

    float3 diffuse = facing * albedo;
    float3 specular = specular_amount * kSpecularStrength;

    // The shadow takes away the sun and nothing else. The ambient term is the light from
    // the rest of the sky, and a crate does not block that.
    return ambient + globals.sun_colour.rgb * (diffuse + specular) *
                         sunlight(input.world_position);
}

[shader("fragment")]
float4 fragmentMain(VSOutput input) : SV_Target {
    float3 tint = push.draws[push.draw_index].colour.rgb;
    float3 albedo = base_colour_map.Sample(input.uv).rgb * tint;
    return float4(shade(input, albedo), 1.0);
}

// ---------------------------------------------------------------------------
// Key 4: the shadow map itself
// ---------------------------------------------------------------------------

struct InsetOutput {
    float4 position : SV_Position;
    [[vk::location(0)]] float2 uv;
};

// 4.3's single triangle that covers the viewport, here a small viewport in the corner.
[shader("vertex")]
InsetOutput insetVertexMain(uint vertex_id : SV_VulkanVertexID) {
    float2 uv = float2((vertex_id << 1) & 2, vertex_id & 2);
    InsetOutput output;
    output.position = float4(uv * 2.0 - 1.0, 0.0, 1.0);
    output.uv = uv;
    return output;
}

// 5.2's decoding curve. Here it is used backwards from the way 5.2 used it: to make
// the swapchain's encode store the depth's own value as the pixel's byte.
float3 srgb_to_linear(float3 encoded) {
    float3 low = encoded / 12.92;
    float3 high = pow((encoded + 0.055) / 1.055, 2.4);
    return select(encoded <= 0.04045, low, high);
}

// The stored depth, as grey: black at the sun's near plane, white at its far plane. The
// floor runs from about 0.3 to 0.67 across the map, because it tilts away from the sun.
// A depth is not an amount of light, so it is decoded first, as 5.2's UNORM emulation
// did; written straight out it would be encoded, and every grey would come out pale.
[shader("fragment")]
float4 insetFragmentMain(InsetOutput input) : SV_Target {
    float depth = shadow_depth.Sample(input.uv).r;
    return float4(srgb_to_linear(depth.xxx), 1.0);
}
