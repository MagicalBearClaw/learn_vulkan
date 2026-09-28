// 4.8 The PCG hash from 4.6, in Slang.
//
// The same three constants and the same shifts as pcg_hash() in main.cpp, so the same
// input gives the same 32 bits on the GPU as on the CPU -- unsigned arithmetic wraps at
// 2^32 in both languages. A module of its own, because nothing about hashing belongs to
// the scene: 4.8's stars use it, and 4.9's compute shader will.
module random;

public uint pcg_hash(uint input) {
    uint state = input * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// A float in [0, 1) from a hash, by the same 24-bit route as the C++ Random::next().
public float unit_float(uint hash) {
    return float(hash >> 8u) * (1.0 / 16777216.0);
}
