#version 450

layout(location = 0) in vec3 frag_colour;

layout(location = 0) out vec4 out_colour;

// The same two blocks, declared identically. A push constant block and a descriptor
// binding are visible to every stage listed in their stageFlags, and each stage that
// uses them declares them again.
layout(set = 0, binding = 0) uniform Globals {
    vec4 ambient;
    float time;
} globals;

layout(push_constant) uniform Push {
    vec2 offset;
    float scale;
    float phase;
} push;

void main() {
    float brightness = 0.6 + 0.4 * sin(globals.time + push.phase);

    // The ambient term comes from the uniform buffer and is identical for all four
    // quads; the brightness differs because each quad pushed its own phase.
    out_colour = vec4(frag_colour * brightness + globals.ambient.rgb, 1.0);
}
