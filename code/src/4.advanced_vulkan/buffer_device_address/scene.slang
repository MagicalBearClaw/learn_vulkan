// 4.5 The scene shader, with every buffer it reads reached through a pointer.
//
// Compare this with 4.4's version. The lighting is identical, the outputs are identical,
// and the first twenty lines are completely different: there is no ConstantBuffer, no
// binding for per-frame data, and no vertex input struct. Everything the shader needs
// arrives as an address in the push constant and is dereferenced here.

// The vertex, as the shader sees it -- and this declaration is now load-bearing in a way
// it never was. A [[vk::location(0)]] float3 used to be a request that the hardware
// convert whatever the vertex attribute description described into a float3. There is no
// conversion step left: these bytes are read straight out of memory, so this struct has
// to match the C++ one exactly.
//
// It does, and not by luck. Slang lays a struct behind a pointer out the way a C++
// compiler would: float3 occupies twelve bytes and the next member starts right after
// it. That is what makes `struct Vertex` here and `struct Vertex` in main.cpp the same
// thirty-two bytes. A uniform buffer would have obeyed std140 instead, rounded each
// float3 up to sixteen, and silently disagreed.
struct Vertex {
    float3 position;
    float3 normal;
    float2 uv;
};

// 4.4's Globals, unchanged. It was a ConstantBuffer then and is a pointer target now,
// which needed no edit at all: a layout that satisfied std140 also satisfies this.
struct Globals {
    float4x4 view;
    float4x4 projection;
    float4x4 view_rotation;
    float4 view_position;
    float4 sun_direction;
    float4 sun_colour;
    float4 ambient;  // rgb ambient, w shininess
};

// 4.4's push-constant block, which has moved into memory. float3x3 rather than the three
// padded float4 it needed as a uniform: thirty-six bytes, matching a glm::mat3 exactly.
struct DrawData {
    float4x4 model;
    float3x3 normal_matrix;
    float4 colour;
};

// The three pointers and the index. A pointer in Slang is written exactly as it is in
// C++, and compiles to SPIR-V's PhysicalStorageBuffer storage class -- the module
// declares the PhysicalStorageBufferAddresses capability and a 64-bit addressing model
// because of these four lines.
struct PushConstants {
    Globals* globals;
    DrawData* draws;
    Vertex* vertices;
    uint draw_index;
};
[[vk::push_constant]] PushConstants push;

// The one descriptor left in the sample. An image cannot be reached by address.
[[vk::binding(0, 0)]] Sampler2D base_colour_map;

struct VSOutput {
    float4 position : SV_Position;
    [[vk::location(0)]] float3 world_position;
    [[vk::location(1)]] float3 normal;
    [[vk::location(2)]] float2 uv;
};

static const float kSpecularStrength = 0.2;
static const float kCutoutThreshold = 0.5;

[shader("vertex")]
VSOutput vertexMain(uint vertex_id : SV_VulkanVertexID) {
    // Vertex fetch, written out. SV_VulkanVertexID is the index the index buffer
    // produced, and the index buffer is still bound through the command buffer -- index
    // fetch is fixed-function hardware that runs before any shader. Vertex fetch is not,
    // any more; it is this line.
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

float3 shade(VSOutput input, float3 albedo, bool two_sided) {
    float3 normal = normalize(input.normal);
    // Reading two fields out of a 256-byte struct through a pointer, rather than binding
    // the whole struct and reading two fields out of it. The generated code is the same
    // shape either way: an address, an offset, a load.
    float3 to_eye = normalize(push.globals.view_position.xyz - input.world_position);
    if (two_sided && dot(normal, to_eye) < 0.0) {
        normal = -normal;
    }

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
    float3 tint = push.draws[push.draw_index].colour.rgb;
    float3 albedo = base_colour_map.Sample(input.uv).rgb * tint;
    return float4(shade(input, albedo, false), 1.0);
}

[shader("fragment")]
float4 cutoutFragmentMain(VSOutput input) : SV_Target {
    float4 sampled = base_colour_map.Sample(input.uv);
    if (sampled.a < kCutoutThreshold) {
        discard;
    }
    float3 tint = push.draws[push.draw_index].colour.rgb;
    return float4(shade(input, sampled.rgb * tint, true), 1.0);
}
