#include "bukkitAllocateRootSignature.hlsl"  // Includes the ROOTSIG definition
#include "../../pbmpmShaders/PBMPMCommon.hlsl"  // Includes the TileDataSize definition

// Taken from https://github.com/electronicarts/pbmpm

// Root constants bound to b0
ConstantBuffer<PBMPMConstants> g_simConstants : register(b0);
// UAVs and SRVs
RWStructuredBuffer<uint> g_bukkitIndirectDispatch : register(u3);
RWStructuredBuffer<uint> g_bukkitParticleAllocator : register(u1);
StructuredBuffer<uint> g_bukkitCounts : register(t0);
RWStructuredBuffer<BukkitThreadData> g_bukkitThreadData : register(u0);
RWStructuredBuffer<uint> g_bukkitIndexStart : register(u2);

// Compute Shader Entry Point
[numthreads(GridDispatchSize, GridDispatchSize, GridDispatchSize)]
void main(uint3 id : SV_DispatchThreadID)
{
    // Ensure the current invocation is within bounds
    if (id.x >= g_simConstants.bukkitCountX || id.y >= g_simConstants.bukkitCountY || id.z >= g_simConstants.bukkitCountZ)
    {
        return;
    }

    // Compute the bukkit index
    uint bukkitIndex = bukkitAddressToIndex(uint3(id.xyz), g_simConstants.bukkitCountX, g_simConstants.bukkitCountY);

    // Get the particle count for the current bukkit
    uint bukkitCount = g_bukkitCounts[bukkitIndex];
    uint bukkitCountResidual = bukkitCount % ParticleDispatchSize;

    // Skip if no particles in the current bukkit
    if (bukkitCount == 0)
    {
        return;
    }

    // Calculate the number of dispatch groups required
    uint dispatchCount = divUp(bukkitCount, ParticleDispatchSize);

    // Update global dispatch and particle allocators atomically
    uint dispatchStartIndex;
    InterlockedAdd(g_bukkitIndirectDispatch[0], dispatchCount, dispatchStartIndex);
    
    uint particleStartIndex;
    InterlockedAdd(g_bukkitParticleAllocator[0], bukkitCount, particleStartIndex);

    // Record the start index for this bukkit
    g_bukkitIndexStart[bukkitIndex] = particleStartIndex;

    // Write thread data for each dispatch group
    for (uint i = 0; i < dispatchCount; i++)
    {
        // Group count is equal to ParticleDispatchSize except for the final dispatch for this
        // bukkit in which case it's equal to the residual count
        uint groupCount = ParticleDispatchSize;
        if (bukkitCountResidual != 0 && i == dispatchCount - 1)
        {
            groupCount = bukkitCountResidual;
        }

        // Write the thread data
        BukkitThreadData data = {particleStartIndex + i * ParticleDispatchSize, groupCount, id.x, id.y, id.z};
        g_bukkitThreadData[i + dispatchStartIndex] = data;
        
    }
}