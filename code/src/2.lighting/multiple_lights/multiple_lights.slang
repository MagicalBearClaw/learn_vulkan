// Multiple lights: six of them, of three different kinds, in one array.
//
// 2.5 shaded with one light whose kind was fixed at pipeline creation. That worked
// because there was exactly one kind to pick. With a sun, four bulbs and a torch in the
// same scene there is no single kind to specialise on, so the kind moves back into the
// data -- one `type` field per light -- and the specialisation constant takes the other
// job: fixing how many lights the loop runs for.
//
// The shading itself has not changed since 2.2. What changed is that it is now written
// once, in shade(), and called from three small functions that differ only in how they
// work out `to_light` and `intensity`.

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

// 2.5's struct with one field added. A directional light uses `direction` and nothing
// else; a point light uses `position` and the attenuation terms; a spotlight uses
// everything. `type` is what makes a single array able to hold all three.
struct Light {
    float4 position;
    float4 direction;
    float4 ambient;
    float4 diffuse;
    float4 specular;

    float constant_term;
    float linear_term;
    float quadratic_term;

    float inner_cutoff;
    float outer_cutoff;

    int type;  // 0 directional, 1 point, 2 spot
};

// The array is sized for the worst case the program will ever ask for. A uniform block
// cannot have a run-time length, and it cannot be sized by a specialisation constant
// either -- the layout has to be known when the block is declared -- so this is a plain
// compile-time constant, and the count below decides how much of it gets read.
static const int kMaxLights = 8;

struct Globals {
    float4x4 view;
    float4x4 projection;
    float4 view_position;
    Light lights[kMaxLights];
};
[[vk::binding(0, 0)]] ConstantBuffer<Globals> globals;

[[vk::binding(0, 1)]] Sampler2D diffuse_map;
[[vk::binding(1, 1)]] Sampler2D specular_map;

struct MaterialParams {
    float shininess;
};
[[vk::binding(2, 1)]] ConstantBuffer<MaterialParams> material;

// The lamp cubes are drawn in their own light's colour, so the push block is read by both
// stages now. model + normal_matrix + colour is exactly 128 bytes, which is all the push
// constant space Vulkan guarantees.
struct PushConstants {
    float4x4 model;
    float3x3 normal_matrix;
    float4 colour;
};
[[vk::push_constant]] PushConstants push;

// How many entries of globals.lights are live. Specialised at pipeline creation, so the
// loop below has a constant trip count and the compiler is free to unroll it -- which it
// could not do if this were a uniform.
[[vk::constant_id(0)]] const int kLightCount = 6;

static const int kDirectional = 0;
static const int kPoint = 1;

// Everything the shading needs to know about the fragment, gathered once instead of being
// recomputed inside each of the six iterations.
struct Surface {
    float3 world_position;
    float3 normal;
    float3 to_eye;
    float3 albedo;
    float3 shine;
};

// The Phong terms from 2.2, unchanged, for a light arriving along `to_light` at strength
// `intensity`.
//
// Ambient is inside the intensity multiplier here, where 2.5 kept it outside. With one
// light, unattenuated ambient was a reasonable stand-in for everything the scene bounces.
// With six it is not: six ambient floors add up, and the scene washes out to flat grey.
// The sun's intensity is 1, so its ambient survives in full and does the job on its own.
float3 shade(Light light, Surface surface, float3 to_light, float intensity) {
    float3 ambient = light.ambient.rgb * surface.albedo;

    float diffuse_amount = max(dot(surface.normal, to_light), 0.0);
    float3 diffuse = light.diffuse.rgb * diffuse_amount * surface.albedo;

    float3 reflected = reflect(-to_light, surface.normal);
    float specular_amount = pow(max(dot(surface.to_eye, reflected), 0.0), material.shininess);
    float3 specular = light.specular.rgb * specular_amount * surface.shine;

    return intensity * (ambient + diffuse + specular);
}

float attenuation_of(Light light, float distance) {
    return 1.0 / (light.constant_term + light.linear_term * distance +
                  light.quadratic_term * distance * distance);
}

float3 directional_contribution(Light light, Surface surface) {
    // Parallel rays: the same direction everywhere, no falloff.
    return shade(light, surface, normalize(-light.direction.xyz), 1.0);
}

float3 point_contribution(Light light, Surface surface) {
    float3 offset = light.position.xyz - surface.world_position;
    float distance = length(offset);
    return shade(light, surface, offset / distance, attenuation_of(light, distance));
}

float3 spot_contribution(Light light, Surface surface) {
    float3 offset = light.position.xyz - surface.world_position;
    float distance = length(offset);
    float3 to_light = offset / distance;

    // A point light restricted to a cone, with the soft edge 2.5 built.
    float theta = dot(-to_light, normalize(light.direction.xyz));
    float cone = saturate((theta - light.outer_cutoff) /
                          (light.inner_cutoff - light.outer_cutoff));

    return shade(light, surface, to_light, attenuation_of(light, distance) * cone);
}

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
    Surface surface;
    surface.world_position = input.world_position;
    surface.normal = normalize(input.normal);
    surface.to_eye = normalize(globals.view_position.xyz - input.world_position);
    surface.albedo = diffuse_map.Sample(input.uv).rgb;
    surface.shine = specular_map.Sample(input.uv).rgb;

    // Light is additive: every light that reaches this fragment contributes, and the
    // contributions simply sum. That is not an approximation -- it is how light behaves,
    // and it is why this loop is the whole of "multiple lights".
    float3 colour = float3(0.0);
    for (int i = 0; i < kLightCount; ++i) {
        Light light = globals.lights[i];

        // This branch is on data, not on a specialisation constant, so it is a real
        // branch. Neighbouring fragments almost always take the same one, which is the
        // case GPUs handle well: a warp only pays for a branch when its lanes disagree.
        switch (light.type) {
        case kDirectional:
            colour += directional_contribution(light, surface);
            break;
        case kPoint:
            colour += point_contribution(light, surface);
            break;
        default:
            colour += spot_contribution(light, surface);
            break;
        }
    }

    return float4(colour, 1.0);
}

[shader("fragment")]
float4 lampFragmentMain(VSOutput input) : SV_Target {
    // Each lamp is drawn in its own light's colour, pushed with the model matrix.
    return push.colour;
}
