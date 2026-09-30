// 4.9 The rocks and the planet: 4.8's, unchanged.
import scene;

// 4.6's pointer path.
[shader("vertex")]
VSOutput vertexMain(uint vertex_id : SV_VulkanVertexID,
                    uint instance_id : SV_VulkanInstanceID) {
    return transform(push.vertices[vertex_id], push.instances[instance_id]);
}

// A rock is its instance's colour, with 4.6's modest highlight.
struct RockMaterial : IMaterial {
    float3 albedo(VSOutput input) { return input.colour; }
    float specular_strength() { return 0.15; }
};

// 4.6's bands, as a material.
struct BandedMaterial : IMaterial {
    float3 albedo(VSOutput input) {
        float height = normalize(input.world_position).y;
        float band = 0.5 + 0.5 * sin(height * 14.0 + 1.3 * sin(height * 5.0));
        return lerp(input.colour, float3(0.93, 0.83, 0.66), band * 0.55);
    }
    float specular_strength() { return 0.15; }
};

[shader("fragment")]
float4 fragmentMain(VSOutput input) : SV_Target {
    RockMaterial material;
    return float4(shade(material, input), 1.0);
}

[shader("fragment")]
float4 planetFragmentMain(VSOutput input) : SV_Target {
    BandedMaterial material;
    return float4(shade(material, input), 1.0);
}
