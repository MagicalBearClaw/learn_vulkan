struct VSInput {
    [[vk::location(0)]] float2 position;
    [[vk::location(1)]] float2 uv;
};

struct VSOutput {
    float4 position : SV_Position;
    [[vk::location(0)]] float2 uv;
};

struct PushConstants {
    float4x4 model;
};
[[vk::push_constant]] PushConstants push;

[[vk::binding(0, 0)]] Sampler2D tex;

[shader("vertex")]
VSOutput vertexMain(VSInput input) {
    VSOutput output;

    // mul(matrix, vector) applies the matrix to the vector: read `mul(model, position)`
    // as "model times position", the same order GLSL writes as `model * position`. The
    // vertex is a column vector on the right, so transforms compose right to left, and
    // that stays true for the whole series.
    //
    // Slang does not use `*` for this. In Slang -- as in HLSL -- `*` between two
    // matrices multiplies them element by element, which compiles happily and is almost
    // never what you meant. Use mul() for anything that is a transform.
    //
    // The matrix arrives as glm's bytes, copied straight out of a glm::mat4 with no
    // transpose on the way. That works because Slang's memory layout and its
    // mul() convention are matched to each other: the two conventions cancel, and the
    // arithmetic comes out the same as GLSL's. There is nothing to compensate for.
    //
    // The position is promoted to four components: z = 0 puts the quad on the near
    // plane, and w = 1 marks it as a *point* rather than a direction. A vector with
    // w = 0 is unaffected by the translation part of a matrix, which is exactly what
    // you want for a surface normal -- chapter 2.2 relies on that.
    output.position = mul(push.model, float4(input.position, 0.0, 1.0));

    output.uv = input.uv;
    return output;
}

[shader("fragment")]
float4 fragmentMain(VSOutput input) : SV_Target {
    return tex.Sample(input.uv);
}
