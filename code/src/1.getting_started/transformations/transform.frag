#version 450

layout(location = 0) in vec2 frag_uv;

layout(location = 0) out vec4 out_colour;

layout(set = 0, binding = 0) uniform sampler2D tex;

void main() {
    out_colour = texture(tex, frag_uv);
}
