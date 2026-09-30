// 4.9 The PCG hash from 4.6, in Slang, and now 4.6's Random stream as well.
//
// The same three constants and the same shifts as pcg_hash() in main.cpp, so the same
// input gives the same 32 bits on the GPU as on the CPU -- unsigned arithmetic wraps at
// 2^32 in both languages. A module of its own, because nothing about hashing belongs to
// the scene: the stars use it, and so does the compute shader.
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

// 4.6's Random class, in Slang: a state that each next() hashes forward. Seeded with
// the same number, it produces the same floats in the same order as the C++ one, which
// is what lets the compute shader rebuild exactly the belt the CPU builds.
public struct Random {
    uint state;

    public __init(uint seed) { state = pcg_hash(seed); }

    // `mutating` because it changes the struct it is called on. Slang methods are
    // read-only unless they say otherwise.
    [mutating]
    public float next() {
        state = pcg_hash(state);
        return unit_float(state);
    }

    [mutating]
    public float range(float low, float high) {
        return low + (high - low) * next();
    }
};
