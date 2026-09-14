// The vertices no longer live here.
//
// In 1.7 this shader had a constant array and indexed it with the vertex index. Now the
// data arrives from a vertex buffer, and this struct is how the shader receives it. The
// [[vk::location(n)]] numbers match the VkVertexInputAttributeDescription entries in
// main.cpp -- that pairing is the entire contract, and getting it wrong is one of the
// few mistakes the validation layers cannot catch for you.
struct VSInput {
    [[vk::location(0)]] float2 position;
    [[vk::location(1)]] float3 colour;
};

// Unchanged from 1.7: the fragment stage neither knows nor cares whether the colour it
// receives came from a constant array in the vertex shader or from GPU memory.
struct VSOutput {
    float4 position : SV_Position;
    [[vk::location(0)]] float3 colour;
};

[shader("vertex")]
VSOutput vertexMain(VSInput input) {
    VSOutput output;
    output.position = float4(input.position, 0.0, 1.0);
    output.colour = input.colour;
    return output;
}

[shader("fragment")]
float4 fragmentMain(VSOutput input) : SV_Target {
    return float4(input.colour, 1.0);
}
