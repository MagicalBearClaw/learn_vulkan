#version 450

// One invocation per pixel covered by the triangle.
//
// frag_colour is not the value any single vertex wrote. The hardware interpolates the
// three vertex outputs across the triangle's surface, weighted by how close the pixel
// is to each corner, which is why the result is a smooth gradient rather than three
// flat regions. That interpolation is free and it is the basis of almost everything
// in the lighting chapters.

layout(location = 0) in vec3 frag_colour;

layout(location = 0) out vec4 out_colour;

void main() {
    out_colour = vec4(frag_colour, 1.0);
}
