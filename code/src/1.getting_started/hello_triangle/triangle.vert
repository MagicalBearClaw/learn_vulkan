#version 450

// The triangle's vertices live in the shader itself.
//
// There is no vertex buffer in this chapter, and that is deliberate: buffers, memory
// and staging are chapter 1.8's subject, and mixing them in here would mean learning
// two large things at once. gl_VertexIndex counts 0, 1, 2 across the three vertices
// that vkCmdDraw asked for, so it can just index a constant array.

layout(location = 0) out vec3 frag_colour;

// Clip space in Vulkan: x and y run -1..1 with y pointing DOWN the screen, and z runs
// 0..1. OpenGL has y pointing up and z running -1..1, which is the single most common
// thing to trip over when porting shaders. Here, -0.5 in y is the top of the screen.
const vec2 positions[3] = vec2[](
    vec2( 0.0, -0.5),   // top
    vec2( 0.5,  0.5),   // bottom right
    vec2(-0.5,  0.5)    // bottom left
);

const vec3 colours[3] = vec3[](
    vec3(1.0, 0.0, 0.0),
    vec3(0.0, 1.0, 0.0),
    vec3(0.0, 0.0, 1.0)
);

void main() {
    gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
    frag_colour = colours[gl_VertexIndex];
}
