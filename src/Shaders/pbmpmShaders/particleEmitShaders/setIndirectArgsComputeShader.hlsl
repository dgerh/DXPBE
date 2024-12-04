#include "setIndirectArgsRootSignature.hlsl"  // Includes the ROOTSIG definition
#include "../../pbmpmShaders/PBMPMCommon.hlsl"  // Includes the TileDataSize definition
// UAVs and SRVs
RWStructuredBuffer<uint> g_simIndirectArgs : register(u0);
RWStructuredBuffer<uint> g_renderIndirectArgs : register(u1);

StructuredBuffer<uint> g_particleCount : register(t0);

// Compute Shader Entry Point
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    g_simIndirectArgs[0] = divUp(g_particleCount[0], ParticleDispatchSize);
    g_renderIndirectArgs[1] = g_particleCount[0];
}