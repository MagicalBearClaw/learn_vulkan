// 4.4 The two things that read an environment map: the sky behind everything, and the
// mirrored sphere in front of it.
//
// They share a file because they share a binding. Both sample a cubemap, and a cubemap
// is not addressed by a uv pair -- it is addressed by a *direction*. Neither of these
// shaders contains a texture coordinate anywhere. The skybox looks up the direction from
// the camera to the vertex; the sphere looks up the direction the world reflects into.
// That is the whole idea of the chapter, and both halves of it are below.

// One input struct per vertex shader, each declaring only the attributes its pipeline
// is given. A shader input with no attribute behind it is as much an error as an
// attribute no shader reads, and the validation layers report both.
struct SkyVSInput {
    [[vk::location(0)]] float3 position;
};

struct MirrorVSInput {
    [[vk::location(0)]] float3 position;
    [[vk::location(1)]] float3 normal;
};

struct Globals {
    float4x4 view;
    float4x4 projection;
    float4x4 view_rotation;
    float4 view_position;
    float4 sun_direction;
    float4 sun_colour;
    float4 ambient;
};
[[vk::binding(0, 0)]] ConstantBuffer<Globals> globals;

// The environment. Declared SamplerCube rather than Sampler2D, and that is the only
// difference in the shader -- the descriptor set layout behind it is byte for byte the
// one the crates and the floor use. What makes this a cubemap is the *view* the
// descriptor points at, which the sample created with VK_IMAGE_VIEW_TYPE_CUBE.
[[vk::binding(0, 1)]] SamplerCube environment;

// The same 128-byte block the scene shader pushes; the mirror reads the last float4 as
// parameters rather than as a colour.
struct PushConstants {
    float4x4 model;
    float3x3 normal_matrix;
    float4 params;  // x: which mode, y: the refraction ratio
};
[[vk::push_constant]] PushConstants push;

static const int kReflect = 0;
static const int kRefract = 1;
static const int kFresnel = 2;

// A very slightly blue mirror. A real one absorbs a little of what it reflects, and a
// perfectly neutral reflection of a blue sky reads as suspiciously bright.
static const float3 kMirrorTint = float3(0.92, 0.95, 1.0);

// ---------------------------------------------------------------------------
// The sky
// ---------------------------------------------------------------------------

struct SkyVSOutput {
    float4 position : SV_Position;
    [[vk::location(0)]] float3 direction;
};

[shader("vertex")]
SkyVSOutput vertexMain(SkyVSInput input) {
    SkyVSOutput output;

    // The skybox is a unit cube centred on the origin, so a vertex's own position *is*
    // the direction from the centre out to it. No texture coordinate is needed or used.
    output.direction = input.position;

    // view_rotation is the camera's view matrix with the translation dropped. The box
    // therefore turns when you turn your head and never moves when you walk, which is
    // what makes it read as infinitely far away rather than as a room you are standing
    // in. Leave the translation in and you can walk out through the wall.
    float4 clip = mul(globals.projection,
                      mul(globals.view_rotation, float4(input.position, 1.0)));

    // Force z = w, so that after the perspective divide the depth is exactly 1.0 -- the
    // furthest value the depth buffer can hold. Against a buffer cleared to 1.0 a LESS
    // test would reject every one of these fragments and the sky would never appear, so
    // this pipeline tests LEQUAL. Drawing the sky last then costs almost nothing: every
    // pixel the scene already covered fails the test before the fragment shader runs.
    output.position = clip.xyww;
    return output;
}

[shader("fragment")]
float4 fragmentMain(SkyVSOutput input) : SV_Target {
    // Interpolating a direction across a triangle does not preserve its length, so it
    // is normalised here. Cubemap face selection only looks at which component is
    // largest and would not care; the mirror below cares, and doing it the same way in
    // both places is worth more than saving one instruction.
    return float4(environment.Sample(normalize(input.direction)).rgb, 1.0);
}

// ---------------------------------------------------------------------------
// The mirrored sphere
// ---------------------------------------------------------------------------

struct MirrorVSOutput {
    float4 position : SV_Position;
    [[vk::location(0)]] float3 world_position;
    [[vk::location(1)]] float3 normal;
};

[shader("vertex")]
MirrorVSOutput mirrorVertexMain(MirrorVSInput input) {
    MirrorVSOutput output;
    float4 world = mul(push.model, float4(input.position, 1.0));
    output.world_position = world.xyz;
    output.position = mul(globals.projection, mul(globals.view, world));
    output.normal = mul(push.normal_matrix, input.normal);
    return output;
}

[shader("fragment")]
float4 mirrorFragmentMain(MirrorVSOutput input) : SV_Target {
    float3 normal = normalize(input.normal);

    // The incident direction: from the eye toward the surface. Both reflect() and
    // refract() are defined for a vector pointing *at* the surface, and getting this
    // the wrong way round is the classic environment-mapping bug -- the reflection
    // still moves when you move, so it looks plausible, and it is mirrored.
    float3 incident = normalize(input.world_position - globals.view_position.xyz);

    float3 reflected = environment.Sample(reflect(incident, normal)).rgb * kMirrorTint;

    int mode = int(push.params.x);
    if (mode == kReflect) {
        return float4(reflected, 1.0);
    }

    // refract() takes the ratio of the two indices of refraction: the medium the light
    // is leaving over the one it is entering. Air into glass is 1.00 / 1.52.
    float3 bent = refract(incident, normal, push.params.y);

    // refract() returns the zero vector when the angle is past the critical one and the
    // light cannot leave at all. Sampling a cubemap with a zero direction has no
    // meaningful answer, so those fragments fall back to the reflection -- which is
    // also the physically right answer, because total internal reflection is what
    // actually happens there.
    float3 refracted = dot(bent, bent) < 1e-6 ? reflected
                                              : environment.Sample(bent).rgb;
    if (mode == kRefract) {
        return float4(refracted, 1.0);
    }

    // Both at once, mixed by Schlick's approximation to the Fresnel term. A surface
    // seen face-on is mostly transparent; the same surface seen edge-on is mostly a
    // mirror. It is why a pond at your feet shows the gravel and the same pond across
    // the park shows the sky.
    float f0 = 0.04;
    float grazing = 1.0 - max(dot(-incident, normal), 0.0);
    float fresnel = f0 + (1.0 - f0) * pow(grazing, 5.0);
    return float4(lerp(refracted, reflected, fresnel), 1.0);
}
