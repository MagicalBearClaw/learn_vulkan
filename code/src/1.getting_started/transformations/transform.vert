#version 450

layout(location = 0) in vec2 in_position;
layout(location = 1) in vec2 in_uv;

layout(location = 0) out vec2 frag_uv;

layout(push_constant) uniform Push {
    mat4 model;
} push;

void main() {
    // The vertex is a column vector on the right, so the matrix is applied to it: read
    // `model * position`, not `position * model`. GLSL has an operator for the other
    // convention too, and mixing them up gives you a transposed transform that looks
    // almost right and never is.
    //
    // The position is promoted to four components: z = 0 puts the quad on the near
    // plane, and w = 1 marks it as a *point* rather than a direction. A vector with
    // w = 0 is unaffected by the translation part of a matrix, which is exactly what
    // you want for a surface normal -- chapter 2.2 relies on that.
    gl_Position = push.model * vec4(in_position, 0.0, 1.0);

    frag_uv = in_uv;
}
