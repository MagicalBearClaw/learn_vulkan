// 4.9 Every rock's matrices for one moment, computed on the GPU, one thread per rock.
//
// This is what the CPU path's rock_at() in main.cpp does, line for line: the same random
// stream, the same orbit, the same tumble, the same two matrices. The constants below
// have to match main.cpp's, and only the pixel comparison in the article checks that
// they do.
import scene;
import random;

static const float kBeltInner = 9.0;
static const float kBeltOuter = 17.0;
static const float kBeltThickness = 0.6;
static const float kRockMinSize = 0.04;
static const float kRockMaxSize = 0.22;
static const float kMaxTumble = 0.5;
static const float kGravity = 72.0;
static const float kTwoPi = 6.28318530718;

// This dispatch's own push constants. The compute pipeline has a layout of its own, and
// the graphics block `scene` declares is never used here, so it is not in this
// module's SPIR-V at all.
struct OrbitPush {
    Instance* instances;  // rock i is written to instances[first + i]
    uint first;
    uint count;
    float time;
};
[[vk::push_constant]] OrbitPush orbit;

// glm::rotate's axis-angle rotation, as its three columns. The article derives it.
void rotation_columns(float3 axis, float angle, out float3 column0,
                      out float3 column1, out float3 column2) {
    float c = cos(angle);
    float s = sin(angle);
    float3 t = (1.0 - c) * axis;
    column0 = float3(c + t.x * axis.x, t.x * axis.y + s * axis.z,
                     t.x * axis.z - s * axis.y);
    column1 = float3(t.y * axis.x - s * axis.z, c + t.y * axis.y,
                     t.y * axis.z + s * axis.x);
    column2 = float3(t.z * axis.x + s * axis.y, t.z * axis.y - s * axis.x,
                     c + t.z * axis.z);
}

Instance rock_at(uint index, float time) {
    Random random = Random(index);

    float radius =
        sqrt(random.range(kBeltInner * kBeltInner, kBeltOuter * kBeltOuter));
    float angle = random.range(0.0, kTwoPi);
    float height = random.range(-0.5, 0.5) * kBeltThickness;
    float3 axis = normalize(float3(random.range(-1.0, 1.0),
                                   random.range(-1.0, 1.0),
                                   random.range(0.1, 1.0)));
    float spin = random.range(0.0, kTwoPi);
    float size = random.range(kRockMinSize, kRockMaxSize);
    float3 scale = size * float3(random.range(0.6, 1.4), random.range(0.6, 1.4),
                                 random.range(0.6, 1.4));
    float grey = random.range(0.35, 0.65);
    float warmth = random.range(0.0, 0.12);
    float tumble = random.range(-kMaxTumble, kMaxTumble);

    // Kepler: a circular orbit's angular speed is sqrt(GM / r^3).
    float speed = sqrt(kGravity / (radius * radius * radius));
    float orbit_angle = angle + speed * time;
    float3 position =
        float3(radius * cos(orbit_angle), height, radius * sin(orbit_angle));

    float3 c0, c1, c2;
    rotation_columns(axis, spin + tumble * time, c0, c1, c2);

    // model = translate * rotate * scale: its columns are the rotation's columns
    // stretched by the scale, then the position. The normal matrix, the inverse
    // transpose of the upper 3x3, is for a rotation times a scale the rotation's
    // columns divided by the scale instead. Slang's constructors take rows, so the
    // columns go in as rows and come out through a transpose, as in 4.6.
    Instance instance;
    instance.model =
        transpose(float4x4(float4(c0 * scale.x, 0.0), float4(c1 * scale.y, 0.0),
                           float4(c2 * scale.z, 0.0), float4(position, 1.0)));
    instance.normal_matrix =
        transpose(float3x3(c0 / scale.x, c1 / scale.y, c2 / scale.z));
    instance.colour = float4(grey + warmth, grey + warmth * 0.5, grey, 1.0);
    return instance;
}

// 64 threads to a workgroup; the article explains the number. The dispatch rounds the
// rock count up to whole workgroups, so the last one can have threads with no rock to
// work on, and they must do nothing.
[shader("compute")]
[numthreads(64, 1, 1)]
void computeMain(uint3 thread : SV_DispatchThreadID) {
    uint rock = thread.x;
    if (rock >= orbit.count) {
        return;
    }
    orbit.instances[orbit.first + rock] = rock_at(rock, orbit.time);
}
