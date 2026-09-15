// Phong lighting: ambient, diffuse and specular, computed per fragment in world space.
//
// The file has the same shape as 2.1's -- one vertex stage, a lit fragment stage and a
// lamp fragment stage -- and the lamp is untouched. Everything new is in what the vertex
// stage hands on and in the lit fragment stage's arithmetic.

struct VSInput {
    [[vk::location(0)]] float3 position;
    [[vk::location(1)]] float3 normal;
};

// The fragment stage now needs to know *where* it is and *which way* the surface faces,
// so both travel alongside the clip-space position. Both are in world space: the light
// and the camera positions are world-space too, and every vector in a dot product has to
// be in the same space or the answer is meaningless.
struct VSOutput {
    float4 position : SV_Position;
    [[vk::location(0)]] float3 world_position;
    [[vk::location(1)]] float3 normal;
};

// Set 0, per frame. Gains the light's position and the camera's position, both as
// float4 rather than float3 so that std140 needs no padding to reason about.
struct Globals {
    float4x4 view;
    float4x4 projection;
    float4 light_colour;
    float4 light_position;  // xyz; w unused
    float4 view_position;   // xyz; w unused
};
[[vk::binding(0, 0)]] ConstantBuffer<Globals> globals;

// Per object, and now exactly at the 128 bytes every Vulkan device guarantees.
//
// normal_matrix is a float3x3, which in a push constant block occupies 48 bytes -- three
// columns, each padded out to 16 -- not the 36 a glm::mat3 takes. main.cpp sends three
// glm::vec4 columns for exactly that reason. See the article; it is a trap.
struct PushConstants {
    float4x4 model;
    float3x3 normal_matrix;
    float4 object_colour;
};
[[vk::push_constant]] PushConstants push;

[shader("vertex")]
VSOutput vertexMain(VSInput input) {
    VSOutput output;

    float4 world = mul(push.model, float4(input.position, 1.0));
    output.world_position = world.xyz;
    output.position = mul(globals.projection, mul(globals.view, world));

    // Not mul(push.model, normal).
    //
    // A normal is not a position and does not transform like one. Stretch a surface
    // along x and its normal has to lean *away* from x, which is the inverse of what the
    // stretch did to the positions. The matrix that gets this right is the inverse
    // transpose of the model matrix's upper 3x3, and it is computed on the CPU: Slang,
    // unlike GLSL, has no inverse(), and inverting a matrix per vertex would be a waste
    // even if it did.
    //
    // Interpolation shortens normals between vertices, so they are renormalised in the
    // fragment stage rather than here.
    output.normal = mul(push.normal_matrix, input.normal);
    return output;
}

[shader("fragment")]
float4 fragmentMain(VSOutput input) : SV_Target {
    float3 normal = normalize(input.normal);

    // Unit vector from this fragment toward the light.
    float3 to_light = normalize(globals.light_position.xyz - input.world_position);

    // Ambient: light that has bounced around the scene so many times it arrives from
    // everywhere equally. Real indirect light is expensive to compute -- that is what
    // Part 6's image-based lighting is for -- so for now it is a constant floor that
    // stops faces turned away from the light going completely black.
    const float kAmbientStrength = 0.1;
    float3 ambient = kAmbientStrength * globals.light_colour.rgb;

    // Diffuse: Lambert's cosine law. A beam of light spreads over more surface the more
    // obliquely it arrives, so the energy per unit area falls with the cosine of the angle
    // between the normal and the light. For unit vectors that cosine is the dot product.
    // Negative means the surface faces away, which is no light, not negative light.
    float diffuse_amount = max(dot(normal, to_light), 0.0);
    float3 diffuse = diffuse_amount * globals.light_colour.rgb;

    // Specular: the highlight, a mirror reflection of the light blurred by roughness.
    //
    // reflect() wants the incoming direction, which points from the light *to* the
    // surface -- hence -to_light. The highlight is brightest when the reflected ray
    // heads straight at the eye, and the exponent sets how quickly it falls off: higher is
    // tighter and shinier. 32 is a plastic-ish middle.
    const float kSpecularStrength = 0.5;
    const float kShininess = 32.0;
    float3 to_eye = normalize(globals.view_position.xyz - input.world_position);
    float3 reflected = reflect(-to_light, normal);
    float specular_amount = pow(max(dot(to_eye, reflected), 0.0), kShininess);
    float3 specular = kSpecularStrength * specular_amount * globals.light_colour.rgb;

    // The three terms add up to the light arriving at the eye from this point, and the
    // object's colour multiplies it exactly as in 2.1. The multiply was right; what was
    // missing was how much light actually reaches each fragment.
    float3 result = (ambient + diffuse + specular) * push.object_colour.rgb;
    return float4(result, 1.0);
}

// The lamp, unchanged from 2.1.
[shader("fragment")]
float4 lampFragmentMain(VSOutput input) : SV_Target {
    return float4(globals.light_colour.rgb, 1.0);
}
