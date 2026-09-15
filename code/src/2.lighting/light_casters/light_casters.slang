// Light casters: three different answers to "where does the light come from?".
//
// 2.2 to 2.4 had one kind of light, a point in space shining equally in every direction
// forever. Real lights differ in two ways that matter to the shading: which direction
// the light arrives from at a given point, and how much of it arrives. This file handles
// three standard kinds, chosen per pipeline by a specialisation constant:
//
//   directional   the sun: one direction everywhere, no position, no falloff
//   point         a bulb: a position, light spreading out and weakening with distance
//   spot          a torch: a point light restricted to a cone, with a soft edge
//
// The Phong terms at the bottom of fragmentMain are 2.4's, unchanged. Everything new
// happens before them, in working out `to_light` and a single intensity multiplier.

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

// One struct describes all three kinds. A directional light ignores `position` and the
// attenuation terms; a point light ignores `direction` and the cone; a spotlight uses
// everything. Carrying unused fields is cheaper than three structs and three layouts, and
// it is what lets 2.6 put lights of different kinds in one array.
struct Light {
    float4 position;   // xyz: point and spot
    float4 direction;  // xyz, the way the light *travels*: directional and spot
    float4 ambient;
    float4 diffuse;
    float4 specular;

    // Attenuation, for point and spot: intensity = 1 / (c + l*d + q*d^2).
    float constant_term;
    float linear_term;
    float quadratic_term;

    // The spotlight's cone, stored as cosines so the shader can compare them directly
    // against a dot product without calling acos() per fragment.
    float inner_cutoff;
    float outer_cutoff;
};

struct Globals {
    float4x4 view;
    float4x4 projection;
    float4 view_position;
    Light light;
};
[[vk::binding(0, 0)]] ConstantBuffer<Globals> globals;

[[vk::binding(0, 1)]] Sampler2D diffuse_map;
[[vk::binding(1, 1)]] Sampler2D specular_map;

struct MaterialParams {
    float shininess;
};
[[vk::binding(2, 1)]] ConstantBuffer<MaterialParams> material;

struct PushConstants {
    float4x4 model;
    float3x3 normal_matrix;
};
[[vk::push_constant]] PushConstants push;

// Which kind of light this pipeline shades with. 0 directional, 1 point, 2 spot.
//
// A specialisation constant, chapter 1.9's mechanism, doing the job it exists for. The
// light's kind does not change while the program runs, so the branch below should cost
// nothing -- and with the value fixed at pipeline creation, the driver can compile away
// the two cases that are never taken. A uniform would work too, and would put a real
// branch in every fragment of every frame.
[[vk::constant_id(0)]] const int kLightType = 2;

static const int kDirectional = 0;
static const int kPoint = 1;

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
    float3 surface = diffuse_map.Sample(input.uv).rgb;
    float3 shine = specular_map.Sample(input.uv).rgb;

    Light light = globals.light;

    // Direction from this fragment toward the light, and how strongly the light arrives.
    float3 to_light;
    float intensity = 1.0;

    switch (kLightType) {
    case kDirectional:
        // So far away that every ray is parallel. The light travels along `direction`,
        // so the way back toward it is the negation -- the same for every fragment.
        to_light = normalize(-light.direction.xyz);
        break;

    case kPoint: {
        float3 offset = light.position.xyz - input.world_position;
        float distance = length(offset);
        to_light = offset / distance;
        intensity = 1.0 / (light.constant_term + light.linear_term * distance +
                           light.quadratic_term * distance * distance);
        break;
    }

    default: {
        // A spotlight is a point light first: position, direction, attenuation.
        float3 offset = light.position.xyz - input.world_position;
        float distance = length(offset);
        to_light = offset / distance;
        float attenuation = 1.0 / (light.constant_term + light.linear_term * distance +
                                   light.quadratic_term * distance * distance);

        // Then the cone. theta is the cosine of the angle between the spot's axis and
        // the ray to this fragment: 1.0 dead centre, falling off toward the edge.
        float theta = dot(-to_light, normalize(light.direction.xyz));

        // Between the inner and outer cutoff the intensity ramps from 1 to 0, which is
        // what gives the pool of light a soft edge instead of a hard circle. Cosines
        // shrink as angles grow, so the inner cutoff is the *larger* number.
        float edge = light.inner_cutoff - light.outer_cutoff;
        float cone = saturate((theta - light.outer_cutoff) / edge);

        intensity = attenuation * cone;
        break;
    }
    }

    float3 normal = normalize(input.normal);
    float3 to_eye = normalize(globals.view_position.xyz - input.world_position);

    // Ambient is not scaled by the cone or by distance. It stands for light that has
    // bounced around the scene, and that does not come from the torch's direction.
    float3 ambient = light.ambient.rgb * surface;

    float diffuse_amount = max(dot(normal, to_light), 0.0);
    float3 diffuse = light.diffuse.rgb * diffuse_amount * surface;

    float3 reflected = reflect(-to_light, normal);
    float specular_amount = pow(max(dot(to_eye, reflected), 0.0), material.shininess);
    float3 specular = light.specular.rgb * specular_amount * shine;

    return float4(ambient + intensity * (diffuse + specular), 1.0);
}

[shader("fragment")]
float4 lampFragmentMain(VSOutput input) : SV_Target {
    return float4(globals.light.specular.rgb, 1.0);
}
