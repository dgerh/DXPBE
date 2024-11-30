#include "bukkitCountRootSignature.hlsl"  // Includes the ROOTSIG definition
#include "../../pbmpmShaders/PBMPMCommon.hlsl"  // Includes the TileDataSize definition

// Taken from https://github.com/electronicarts/pbmpm

// Root constants bound to b0
ConstantBuffer<PBMPMConstants> g_simConstants : register(b0);
// CBV, UAVs and SRVs
StructuredBuffer<uint> g_particleCount : register(t0);
StructuredBuffer<Particle> g_particles : register(t1);
RWStructuredBuffer<uint> g_bukkitCounts : register(u0);

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

    // Atomically increment the bukkit count
    InterlockedAdd(g_bukkitCounts[bukkitIndex], 1);
}