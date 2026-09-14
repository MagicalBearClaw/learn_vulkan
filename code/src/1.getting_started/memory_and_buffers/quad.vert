#version 450

// The vertices no longer live here.
//
// In 1.7 this shader had a constant array and indexed it with gl_VertexIndex. Now the
// data arrives from a vertex buffer, and these `in` declarations are how the shader
// receives it. The `location` numbers match the VkVertexInputAttributeDescription
// entries in main.cpp -- that pairing is the entire contract, and getting it wrong is
// one of the few mistakes the validation layers cannot catch for you.

layout(location = 0) in vec2 in_position;
layout(location = 1) in vec3 in_colour;

layout(location = 0) out vec3 frag_colour;

void main() {
    gl_Position = vec4(in_position, 0.0, 1.0);
    frag_colour = in_colour;
}
