// 4.5 The sky and the mirrored sphere, reading their vertices and their frame constants
// through pointers.
//
// 4.4 explains what these two shaders do and why a sky and a mirror are the same lookup
// with different vectors. What is different here is only the plumbing, and the skybox
// shows it at its plainest: it reads one field out of one vertex, and the pipeline it
// belongs to declares no vertex state whatsoever.

struct Vertex {
    float3 position;
    float3 normal;
    float2 uv;
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

[[vk::binding(0, 0)]] SamplerCube environment;

static const float3 kMirrorTint = float3(0.92, 0.95, 1.0);

// ---------------------------------------------------------------------------
// The sky
// ---------------------------------------------------------------------------

struct SkyVSOutput {
    float4 position : SV_Position;
    [[vk::location(0)]] float3 direction;
};

[shader("vertex")]
SkyVSOutput vertexMain(uint vertex_id : SV_VulkanVertexID) {
    // The only field of the vertex this shader wants. In 4.4 the pipeline had to be
    // given exactly one vertex attribute to avoid the validation layers complaining
    // about the two it did not consume; here the question cannot come up, because the
    // pipeline describes no vertex layout at all and the shader takes what it likes.
    float3 position = push.vertices[vertex_id].position;

    SkyVSOutput output;
    output.direction = position;

    float4 clip = mul(push.globals.projection,
                      mul(push.globals.view_rotation, float4(position, 1.0)));
    // z = w, so depth is exactly 1.0 after the divide. See 4.4.
    output.position = clip.xyww;
    return output;
}

[shader("fragment")]
float4 fragmentMain(SkyVSOutput input) : SV_Target {
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
MirrorVSOutput mirrorVertexMain(uint vertex_id : SV_VulkanVertexID) {
    Vertex vertex = push.vertices[vertex_id];
    DrawData draw = push.draws[push.draw_index];

    MirrorVSOutput output;
    float4 world = mul(draw.model, float4(vertex.position, 1.0));
    output.world_position = world.xyz;
    output.position = mul(push.globals.projection, mul(push.globals.view, world));
    output.normal = mul(draw.normal_matrix, vertex.normal);
    return output;
}

[shader("fragment")]
float4 mirrorFragmentMain(MirrorVSOutput input) : SV_Target {
    float3 normal = normalize(input.normal);
    // From the eye toward the surface: reflect() is defined for a vector pointing at
    // the surface, and the classic bug is getting this the wrong way round.
    float3 incident =
        normalize(input.world_position - push.globals.view_position.xyz);
    return float4(environment.Sample(reflect(incident, normal)).rgb * kMirrorTint, 1.0);
}
