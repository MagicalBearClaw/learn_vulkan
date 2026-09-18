// 4.2 Blending and culling: four fragment shaders that barely differ.
//
// Like 4.1, most of this chapter is pipeline state rather than shader code. Culling
// happens before the fragment shader runs; blending happens after it has returned. The
// shader never sees either one.
//
// There is one exception, and it is the reason this file has four entry points instead
// of two: `discard`. Alpha testing -- throwing a fragment away because its texture says
// that texel is empty -- has no fixed-function switch in Vulkan. It is a branch in the
// fragment shader, and it is the only part of this chapter's subject that is written in
// Slang at all.

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

// Set 0, per frame. 4.1's block, unchanged.
struct Globals {
    float4x4 view;
    float4x4 projection;
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
    float4 colour;  // the glass panes carry their transparency in .a
};
[[vk::push_constant]] PushConstants push;

static const float kSpecularStrength = 0.2;

// Anything below this much alpha is not part of the leaf. 0.5 is the conventional
// choice and it is arbitrary: the article works through what moving it does.
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

// `two_sided` is what a quad drawn with culling off needs. Its single normal points one
// way; seen from behind, every lighting term built from that normal is wrong. Flipping
// it to face the viewer is the cheap fix, and it is only ever correct for geometry that
// really is a flat sheet -- which is exactly the geometry culling has to be turned off
// for.
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

// The opaque surfaces: the floor and the crates. 4.1's shader, with the two-sided flag
// off because a closed box and a floor are never seen from behind.
[shader("fragment")]
float4 fragmentMain(VSOutput input) : SV_Target {
    float3 albedo = base_colour_map.Sample(input.uv).rgb * push.colour.rgb;
    return float4(shade(input, albedo, false), 1.0);
}

// The ferns, as a cutout.
//
// `discard` ends the invocation: no colour is written, and -- the part that matters --
// no depth is written either. The quad's empty texels leave the depth buffer untouched,
// so whatever is behind them is still drawn. That is what lets an alpha-tested plant be
// treated as ordinary opaque geometry, needing no sorting and no blending.
//
// The cost is that the decision is binary. A texel is in or out, so the silhouette is
// as jagged as the resolution allows, and the sampler's smooth gradient at the leaf's
// edge is thrown away. 4.9 comes back to this with alpha-to-coverage.
[shader("fragment")]
float4 cutoutFragmentMain(VSOutput input) : SV_Target {
    float4 sampled = base_colour_map.Sample(input.uv);
    if (sampled.a < kCutoutThreshold) {
        discard;
    }
    return float4(shade(input, sampled.rgb * push.colour.rgb, true), 1.0);
}

// The same ferns, handed to the blender instead. No discard, and the sampled alpha goes
// out in the result for the blend equation to use.
//
// This is the version that looks right in a still frame and wrong the moment two plants
// overlap, because blending is order-dependent and these are not sorted. Toggle 4 in the
// sample switches between this entry point and the one above.
[shader("fragment")]
float4 foliageBlendFragmentMain(VSOutput input) : SV_Target {
    float4 sampled = base_colour_map.Sample(input.uv);
    return float4(shade(input, sampled.rgb * push.colour.rgb, true), sampled.a);
}

// The glass panes. Flat colour, alpha and all, straight from the push block: the point
// of the panes is the blend state behind them, and a texture would only get in the way.
[shader("fragment")]
float4 flatFragmentMain(VSOutput input) : SV_Target {
    return push.colour;
}
