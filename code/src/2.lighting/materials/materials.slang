// Materials: the same Phong model as 2.2, with the constants turned into data.
//
// In 2.2 the surface's response to light was baked into the shader -- a single object
// colour, an ambient strength of 0.1, a specular strength of 0.5, a shininess of 32.
// Real surfaces differ in all four, independently, so they become a Material and move
// out of the shader and into a descriptor set of their own.

struct VSInput {
    [[vk::location(0)]] float3 position;
    [[vk::location(1)]] float3 normal;
};

struct VSOutput {
    float4 position : SV_Position;
    [[vk::location(0)]] float3 world_position;
    [[vk::location(1)]] float3 normal;
};

// A light has the same three components as a material, for the same reason: they are
// separate physical effects, and a light can be strong in one and weak in another. A
// dim ambient with a bright diffuse is a single lamp in a dark room; the reverse is an
// overcast day.
struct Light {
    float4 position;   // xyz; w unused
    float4 ambient;
    float4 diffuse;
    float4 specular;
};

// Set 0: per frame. Written once at the top of the frame, bound once.
struct Globals {
    float4x4 view;
    float4x4 projection;
    float4 view_position;  // xyz; w unused
    Light light;
};
[[vk::binding(0, 0)]] ConstantBuffer<Globals> globals;

// Set 1: per material. One descriptor set per material, rebound whenever the material
// changes -- and binding set 1 leaves set 0 exactly where it was.
//
// Each colour says how much of each component of the incoming light the surface sends
// back: `diffuse` is what you would call the surface's colour, `ambient` is usually the
// same hue darker, and `specular` is the colour of the highlight -- white-ish for
// plastics, tinted for metals, which is most of why gold looks like gold.
struct Material {
    float4 ambient;
    float4 diffuse;
    float4 specular;
    float shininess;
};
[[vk::binding(0, 1)]] ConstantBuffer<Material> material;

// Per object. The object colour has gone -- it was a material property all along -- which
// leaves 16 of the 128 guaranteed bytes free again.
struct PushConstants {
    float4x4 model;
    float3x3 normal_matrix;
};
[[vk::push_constant]] PushConstants push;

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
    float3 normal = normalize(input.normal);
    float3 to_light = normalize(globals.light.position.xyz - input.world_position);
    float3 to_eye = normalize(globals.view_position.xyz - input.world_position);

    // The arithmetic is 2.2's, term for term. What changed is that every constant is now
    // a product of what the light emits and what the material reflects, per component.
    float3 ambient = globals.light.ambient.rgb * material.ambient.rgb;

    float diffuse_amount = max(dot(normal, to_light), 0.0);
    float3 diffuse = globals.light.diffuse.rgb * (diffuse_amount * material.diffuse.rgb);

    float3 reflected = reflect(-to_light, normal);
    float specular_amount = pow(max(dot(to_eye, reflected), 0.0), material.shininess);
    float3 specular = globals.light.specular.rgb * (specular_amount * material.specular.rgb);

    return float4(ambient + diffuse + specular, 1.0);
}

// The lamp draws at the light's specular colour: the full, unscaled colour of what the
// light emits, which is what you see when you look straight at it.
[shader("fragment")]
float4 lampFragmentMain(VSOutput input) : SV_Target {
    return float4(globals.light.specular.rgb, 1.0);
}
