// 4.3 The second pass: one triangle, one texture fetch, and whatever you like in between.
//
// This shader never sees a vertex buffer. Three vertices are generated from the vertex
// index alone -- the same trick 1.7's very first triangle used, put to work -- and they
// describe one triangle large enough to cover the whole screen. A quad would do the same
// job with two triangles and a seam down the diagonal where the two meet; one oversized
// triangle has no seam, and the parts that hang off the screen cost nothing because the
// rasteriser clips them.
//
// Everything interesting is in the fragment shader, and it is interesting only because
// of what it is allowed to read: the finished scene, as a texture.

struct VSOutput {
    float4 position : SV_Position;
    [[vk::location(0)]] float2 uv;
};

// The image the scene was rendered into, bound exactly like any other texture. That is
// the whole point of the chapter: by the time this shader runs, the render target from
// the first pass is just a Sampler2D.
[[vk::binding(0, 0)]] Sampler2D scene;

struct PushConstants {
    // The size of one texel of `scene` in uv units -- 1/width, 1/height. The kernels
    // below step by this, so they follow the render scale instead of assuming the
    // scene image is the same size as the window. It is not, in this sample.
    float2 texel;
    int effect;
    int padding;
};
[[vk::push_constant]] PushConstants push;

static const int kNone = 0;
static const int kGrayscale = 1;
static const int kInvert = 2;
static const int kSharpen = 3;
static const int kBlur = 4;
static const int kEdges = 5;
static const int kAlpha = 6;

// Rec. 709 luminance. Green dominates because the eye does: an equal-weight average
// makes a red and a green of the same brightness come out the same grey, which no
// photograph of them ever would.
static const float3 kLuminance = float3(0.2126, 0.7152, 0.0722);

[shader("vertex")]
VSOutput vertexMain(uint vertex_index: SV_VulkanVertexID) {
    // index 0 -> (0,0), 1 -> (2,0), 2 -> (0,2). Doubling in uv means doubling in
    // clip space, so the triangle's two far corners land at +3 and the screen's
    // -1..1 square sits in the corner of it.
    VSOutput output;
    output.uv = float2((vertex_index << 1) & 2, vertex_index & 2);
    output.position = float4(output.uv * 2.0 - 1.0, 0.0, 1.0);
    return output;
}

// Nine samples in a square around `uv`, weighted and summed. Convolution, and the only
// reason the post-processing pass needs a texture rather than a copy: a fragment here
// reads its neighbours, which a fragment in the scene pass could not do.
float3 convolve(float2 uv, float weights[9]) {
    float3 total = float3(0.0);
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            float2 offset = float2(float(x), float(y)) * push.texel;
            total += scene.Sample(uv + offset).rgb * weights[(y + 1) * 3 + (x + 1)];
        }
    }
    return total;
}

[shader("fragment")]
float4 fragmentMain(VSOutput input) : SV_Target {
    float4 sampled = scene.Sample(input.uv);

    if (push.effect == kGrayscale) {
        return float4(float3(dot(sampled.rgb, kLuminance)), 1.0);
    }
    if (push.effect == kInvert) {
        return float4(1.0 - sampled.rgb, 1.0);
    }
    // The scene image's alpha, drawn as a greyscale picture. 4.2 went to some trouble
    // to keep this channel at 1 everywhere; this is what checks that it worked.
    if (push.effect == kAlpha) {
        return float4(float3(sampled.a), 1.0);
    }

    if (push.effect == kSharpen) {
        // Sums to 1, so a flat region comes through unchanged and only differences
        // from the neighbourhood are amplified.
        float weights[9] = {-1, -1, -1, -1, 9, -1, -1, -1, -1};
        return float4(convolve(input.uv, weights), 1.0);
    }
    if (push.effect == kBlur) {
        float weights[9] = {1.0 / 16, 2.0 / 16, 1.0 / 16, 2.0 / 16, 4.0 / 16,
                            2.0 / 16, 1.0 / 16, 2.0 / 16, 1.0 / 16};
        return float4(convolve(input.uv, weights), 1.0);
    }
    if (push.effect == kEdges) {
        // Sums to 0, so a flat region comes out black and only the places where the
        // image changes survive. This is a Laplacian, and it is why the ferns light
        // up and the floor does not.
        float weights[9] = {1, 1, 1, 1, -8, 1, 1, 1, 1};
        return float4(convolve(input.uv, weights), 1.0);
    }

    // kNone. Still a full pass over every pixel, still a texture fetch per fragment --
    // the sample keeps it rather than skipping the pass, so that switching effects
    // changes one number and nothing else.
    return float4(sampled.rgb, 1.0);
}
