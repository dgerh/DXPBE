#include "bukkitInsertRootSignature.hlsl"  // Includes the ROOTSIG definition
#include "../../pbmpmShaders/PBMPMCommon.hlsl"  // Includes the TileDataSize definition

// Taken from https://github.com/electronicarts/pbmpm

// Root constants bound to b0
ConstantBuffer<PBMPMConstants> g_simConstants : register(b0);
// UAVs and SRVs;
StructuredBuffer<Particle> g_particles : register(t0);
StructuredBuffer<uint> g_particleCount : register(t1);
StructuredBuffer<uint> g_bukkitIndexStart : register(t2);
RWStructuredBuffer<uint> g_particleInsertCounters : register(u0);
RWStructuredBuffer<uint> g_particleData : register(u1);

// Compute Shader Entry Point
[numthreads(ParticleDispatchSize, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    // Check if the particle index is out of bounds
    if (id.x >= g_particleCount[0])
    {
        return;
    }

    // Load the particle
    Particle particle = g_particles[id.x];

    // Skip if the particle is disabled
    if (particle.enabled == 0.0f)
    {
        return;
    }

    // Get particle position
    float3 position = particle.position;

    // Calculate the bukkit ID for this particle
    int3 particleBukkit = positionToBukkitId(position);

    // Check if the particle is out of bounds
    if (any(particleBukkit < int3(0, 0, 0)) ||
        uint(particleBukkit.x) >= g_simConstants.bukkitCountX ||
        uint(particleBukkit.y) >= g_simConstants.bukkitCountY ||
        uint(particleBukkit.z) >= g_simConstants.bukkitCountZ)
    {
        return;
    }

    // Calculate the linear bukkit index
    uint bukkitIndex = bukkitAddressToIndex(uint3(particleBukkit), g_simConstants.bukkitCountX, g_simConstants.bukkitCountY);
    // Getting the first particle of this bucket
    uint bukkitIndexStart = g_bukkitIndexStart[bukkitIndex];

    // Atomically increment the particle insert counter
    // (Adds how many particles are in each bukkit again?)
    uint particleInsertCounter;
    InterlockedAdd(g_particleInsertCounters[bukkitIndex], 1, particleInsertCounter);

    // Write the particle index to the particle data buffer
    g_particleData[particleInsertCounter + bukkitIndexStart] = id.x;
}