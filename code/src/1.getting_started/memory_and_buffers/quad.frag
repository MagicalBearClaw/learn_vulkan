#version 450

// Unchanged from 1.7. The fragment shader neither knows nor cares whether the colour
// it receives came from a constant array in the vertex shader or from GPU memory.

layout(location = 0) in vec3 frag_colour;

layout(location = 0) out vec4 out_colour;

void main() {
    out_colour = vec4(frag_colour, 1.0);
}
