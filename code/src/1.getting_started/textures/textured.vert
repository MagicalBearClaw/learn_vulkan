#version 450

layout(location = 0) in vec2 in_position;
layout(location = 1) in vec2 in_uv;

layout(location = 0) out vec2 frag_uv;

layout(push_constant) uniform Push {
    vec2 offset;
    float scale;
    float uv_scale;
} push;

void main() {
    gl_Position = vec4(in_position * push.scale + push.offset, 0.0, 1.0);

    // Multiplying the texture coordinate is how tiling works: uv_scale of 4 makes the
    // coordinate run 0..4 across the quad, and what happens beyond 1 is the sampler's
    // address mode -- REPEAT here, so it wraps four times.
    frag_uv = in_uv * push.uv_scale;
}
