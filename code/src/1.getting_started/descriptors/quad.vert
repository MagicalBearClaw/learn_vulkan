#version 450

layout(location = 0) in vec2 in_position;
layout(location = 1) in vec3 in_colour;

layout(location = 0) out vec3 frag_colour;

// A uniform buffer, reached through descriptor set 0, binding 0.
//
// `set` and `binding` pair with the VkDescriptorSetLayoutBinding in main.cpp and with
// the set index passed to vkCmdBindDescriptorSets. The block's name (Globals) is not
// matched against anything on the host; only the numbers are.
layout(set = 0, binding = 0) uniform Globals {
    vec4 ambient;
    float time;
} globals;

// The push constant block. There is exactly one per pipeline layout -- not one per
// binding -- so everything a draw needs goes in this single struct.
layout(push_constant) uniform Push {
    vec2 offset;
    float scale;
    float phase;
} push;

void main() {
    // A gentle pulse so the two data paths are both visibly live: `time` is shared by
    // all four quads and comes from the uniform buffer, `phase` differs per quad and
    // comes from a push constant.
    float pulse = 0.85 + 0.15 * sin(globals.time + push.phase);

    gl_Position = vec4(in_position * push.scale * pulse + push.offset, 0.0, 1.0);
    frag_colour = in_colour;
}
