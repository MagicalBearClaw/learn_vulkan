// 5.2 The scene shader, lit by three lamps, with its output made to look like a UNORM
// swapchain's on request.
//
// The vertex shader is 5.1's. The lighting is 5.1's Blinn-Phong, summed over an array of
// lamps. What is new is at the end of the fragment shader: what happens to the colour on
// its way out.

struct Vertex {
    float3 position;
    float3 normal;
    float2 uv;
};

static const uint kLampCount = 3;

struct Lamp {
    float4 position;
    float4 colour;
};

// Matches Globals in main.cpp, 260 bytes, scalar layout.
struct Globals {
    float4x4 view;
    float4x4 projection;
    float4 view_position;
    float4 ambient;
    Lamp lamps[kLampCount];
    uint flags;
};

static const uint kEncodeOutput = 1u << 0;
static const uint kShowClipping = 1u << 1;

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

// Blinn-Phong's exponent for a Phong shininess of 16, matched the way 5.1 showed.
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

// 5.1's shade(), for one lamp, without the ambient.
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

    // Light adds. This sum is the reason the shader has to work in linear light: it is
    // only true of amounts of light, not of the encoded numbers that stand for them.
    float3 total = globals.ambient.rgb * albedo;
    for (uint i = 0; i < kLampCount; ++i) {
        total += lamp_light(globals.lamps[i], input.world_position, normal, to_eye, albedo);
    }
    return total;
}

// sRGB's decoding curve, from 2.4: an encoded value in, linear light out.
float3 srgb_to_linear(float3 encoded) {
    float3 low = encoded / 12.92;
    float3 high = pow((encoded + 0.055) / 1.055, 2.4);
    return select(encoded <= 0.04045, low, high);
}

// The last thing every fragment does. With kEncodeOutput set this does nothing, and the
// sRGB swapchain encodes the value as it is written -- what every chapter has done.
//
// Without it, the sample shows what a UNORM swapchain would: the value clamped, stored as
// it is, and then decoded by the display, which assumes everything it is sent is sRGB.
// Writing srgb_to_linear(value) through the sRGB swapchain stores the same bytes, to
// within one step of rounding, because the swapchain's encode undoes the decode.
float4 present(float3 colour) {
    uint flags = push.globals.flags;
    if ((flags & kShowClipping) != 0 && any(colour > 1.0)) {
        return float4(1.0, 0.0, 1.0, 1.0);
    }
    if ((flags & kEncodeOutput) == 0) {
        colour = srgb_to_linear(saturate(colour));
    }
    return float4(colour, 1.0);
}

[shader("fragment")]
float4 fragmentMain(VSOutput input) : SV_Target {
    float3 tint = push.draws[push.draw_index].colour.rgb;
    float3 albedo = base_colour_map.Sample(input.uv).rgb * tint;
    return present(shade(input, albedo));
}

// The lamp itself: its colour, unlit, presented the same way as everything else.
[shader("fragment")]
float4 lampFragmentMain(VSOutput input) : SV_Target {
    return present(push.draws[push.draw_index].colour.rgb);
}
