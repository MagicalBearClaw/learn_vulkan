// 4.6 Instancing: one mesh, drawn many times, each copy reading its own transform.
//
// Everything here is 4.5's scene shader with one new input: the instance index. There
// are two vertex entry points because the chapter compares two ways of getting a copy's
// data into the shader -- fetched by the fixed-function vertex input stage, or read
// through a pointer the shader indexes itself. They produce the same picture, to the
// pixel; the article shows the comparison.

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

// Instances read through a pointer. SV_VulkanInstanceID is Vulkan's InstanceIndex, which
// counts from the draw's firstInstance and not from zero -- the same distinction
// SV_VulkanVertexID makes for vertices. The sample relies on it: the naive path draws one
// instance at a time and says which one through firstInstance.
[shader("vertex")]
VSOutput vertexMain(uint vertex_id : SV_VulkanVertexID,
                    uint instance_id : SV_VulkanInstanceID) {
    return transform(push.vertices[vertex_id], push.instances[instance_id]);
}

// The same Instance, delivered by the vertex input stage instead. Every field arrives as
// an attribute, and a matrix is not an attribute: a float4x4 takes four locations, one
// per column, and a float3x3 takes three.
struct InstanceAttributes {
    [[vk::location(0)]] float4 model0;
    [[vk::location(1)]] float4 model1;
    [[vk::location(2)]] float4 model2;
    [[vk::location(3)]] float4 model3;
    [[vk::location(4)]] float3 normal0;
    [[vk::location(5)]] float3 normal1;
    [[vk::location(6)]] float3 normal2;
    [[vk::location(7)]] float4 colour;
};

[shader("vertex")]
VSOutput attributeVertexMain(uint vertex_id : SV_VulkanVertexID,
                             InstanceAttributes attributes) {
    // Each attribute is one column of a glm matrix. Slang's matrix constructor takes
    // rows, so the columns go in as rows and the transpose puts them back.
    Instance instance;
    instance.model = transpose(float4x4(attributes.model0, attributes.model1,
                                        attributes.model2, attributes.model3));
    instance.normal_matrix = transpose(
        float3x3(attributes.normal0, attributes.normal1, attributes.normal2));
    instance.colour = attributes.colour;
    return transform(push.vertices[vertex_id], instance);
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
