// 4.8 A procedural starfield behind everything, from one triangle.
import scene;
import random;

struct StarsOutput {
    float4 position : SV_Position;
    [[vk::location(0)]] float2 ndc;
};

// 4.3's fullscreen triangle, placed on the far plane: z = w = 1 is depth 1.0, which the
// depth test lets through only where nothing nearer was drawn.
[shader("vertex")]
StarsOutput starsVertexMain(uint vertex_id : SV_VulkanVertexID) {
    float2 uv = float2((vertex_id << 1) & 2, vertex_id & 2);
    StarsOutput output;
    output.ndc = uv * 2.0 - 1.0;
    output.position = float4(output.ndc, 1.0, 1.0);
    return output;
}

// The space around the eye is cut into cubes 1/kCells on a side. A unit direction lands
// in exactly one of them, and each cube holds a star with probability kStarChance.
static const float kCells = 240.0;
static const float kStarChance = 0.08;

[shader("fragment")]
float4 starsFragmentMain(StarsOutput input) : SV_Target {
    // The direction through this pixel, in view space and then in world space. For a
    // symmetric perspective projection, NDC x and y are the view-space direction's x/-z
    // and y/-z scaled by the projection's two diagonal terms, so dividing by them undoes
    // it. The view matrix's rotation is orthonormal, so its inverse is its transpose.
    float4x4 projection = push.globals.projection;
    float3 view_direction =
        float3(input.ndc.x / projection[0][0], input.ndc.y / projection[1][1], -1.0);
    float3 direction =
        normalize(mul(transpose((float3x3)push.globals.view), view_direction));

    // The cell this direction lands in, and that cell's star, if it has one.
    int3 cell = int3(floor(direction * kCells));
    uint seed = pcg_hash(uint(cell.x) ^ pcg_hash(uint(cell.y) ^ pcg_hash(uint(cell.z))));
    if (unit_float(seed) > kStarChance) {
        return float4(0.0, 0.0, 0.0, 0.0);
    }
    uint h1 = pcg_hash(seed);
    uint h2 = pcg_hash(h1);
    uint h3 = pcg_hash(h2);
    float3 centre = normalize((float3(cell) +
                               float3(unit_float(h1), unit_float(h2), unit_float(h3))) /
                              kCells);

    // Brightness falls off with the angle from the star's centre, measured in cells.
    float distance = length(direction - centre) * kCells;
    float brightness = unit_float(pcg_hash(h3));
    float glow = saturate(1.0 - distance * 3.0) *
                 (0.15 + 0.85 * brightness * brightness * brightness);
    return float4(glow.xxx, 1.0);
}
