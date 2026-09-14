#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec2 in_uv;

layout(location = 0) out vec2 frag_uv;

layout(set = 0, binding = 0) uniform Globals {
    mat4 view;
    mat4 projection;
} globals;

layout(push_constant) uniform Push {
    mat4 model;
} push;

void main() {
    // The full chain, read right to left as always:
    //
    //   model space  --model-->  world space
    //                --view-->   view space   (camera at the origin, looking down -z)
    //                --proj-->   clip space   (what gl_Position holds)
    //
    // The hardware then divides by w to get normalised device coordinates and maps
    // those onto the viewport. That divide is what makes distant things small: the
    // projection matrix puts the distance to the camera into w, so a vertex twice as
    // far away is divided by twice as much.
    //
    // Multiplying the three matrices together on the CPU, once per object, would save
    // the GPU two matrix multiplies per vertex. Worth doing when there are many
    // vertices; kept separate here because the three stages are the lesson.
    gl_Position = globals.projection * globals.view * push.model * vec4(in_position, 1.0);

    frag_uv = in_uv;
}
