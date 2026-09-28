// 4.7 Anti-aliasing: 4.6's shader, less the instance-attribute entry point.
//
// Multisampling changes nothing in here. Coverage is decided by the rasteriser, the
// resolve by the attachment, and the only shader-visible difference -- whether this
// runs once per pixel or once per sample -- is chosen by the pipeline, not the source.

// 4.5's vertex, read through a pointer.
struct Vertex {
    float3 position;
    float3 normal;
    float2 uv;
};

// 4.5's Globals with the cube-face field swapped for the belt's rotation. Still 256
// bytes.
struct Globals {
    float4x4 view;
    float4x4 projection;
    float4x4 belt;
    float4 view_position;
    float4 sun_direction;
    float4 sun_colour;
    float4 ambient;  // rgb ambient, w shininess
};

// 4.5's DrawData under a new name, because what it describes changed: not one draw
// call's object, but one copy of a mesh. Same fields, same 116 bytes.
struct Instance {
    float4x4 model;
    float3x3 normal_matrix;
    float4 colour;
};

struct PushConstants {
    Globals* globals;
    Instance* instances;
    Vertex* vertices;
};
[[vk::push_constant]] PushConstants push;

struct VSOutput {
    float4 position : SV_Position;
    [[vk::location(0)]] float3 world_position;
    [[vk::location(1)]] float3 normal;
    [[vk::location(2)]] float3 colour;
};

static const float kSpecularStrength = 0.15;

VSOutput transform(Vertex vertex, Instance instance) {
    // The belt's rotation is applied after the instance's own transform, so every copy
    // turns about the planet's axis together.
    float4x4 model = mul(push.globals.belt, instance.model);

    VSOutput output;
    float4 world = mul(model, float4(vertex.position, 1.0));
    output.world_position = world.xyz;
    output.position = mul(push.globals.projection, mul(push.globals.view, world));
    // The belt is a pure rotation, so it is its own normal matrix.
    output.normal =
        mul((float3x3)push.globals.belt, mul(instance.normal_matrix, vertex.normal));
    output.colour = instance.colour.rgb;
    return output;
}

// 4.6's pointer path. SV_VulkanInstanceID counts from the draw's firstInstance.
[shader("vertex")]
VSOutput vertexMain(uint vertex_id : SV_VulkanVertexID,
                    uint instance_id : SV_VulkanInstanceID) {
    return transform(push.vertices[vertex_id], push.instances[instance_id]);
}

float3 shade(VSOutput input, float3 albedo) {
    float3 normal = normalize(input.normal);
    float3 to_eye = normalize(push.globals.view_position.xyz - input.world_position);
    float3 to_sun = normalize(-push.globals.sun_direction.xyz);

    float diffuse_amount = max(dot(normal, to_sun), 0.0);
    float3 diffuse = push.globals.sun_colour.rgb * diffuse_amount * albedo;

    float3 reflected = reflect(-to_sun, normal);
    float specular_amount =
        pow(max(dot(to_eye, reflected), 0.0), push.globals.ambient.w);
    float3 specular = push.globals.sun_colour.rgb * specular_amount * kSpecularStrength;

    return push.globals.ambient.rgb * albedo + diffuse + specular;
}

[shader("fragment")]
float4 fragmentMain(VSOutput input) : SV_Target {
    return float4(shade(input, input.colour), 1.0);
}

// The planet has no texture. Its bands are a function of latitude, and for a sphere
// centred on the origin the normalised position's y is the sine of the latitude --
// which serves just as well, since all that matters is that it is constant along a
// parallel.
[shader("fragment")]
float4 planetFragmentMain(VSOutput input) : SV_Target {
    float height = normalize(input.world_position).y;
    float band = 0.5 + 0.5 * sin(height * 14.0 + 1.3 * sin(height * 5.0));
    float3 albedo = lerp(input.colour, float3(0.93, 0.83, 0.66), band * 0.55);
    return float4(shade(input, albedo), 1.0);
}
