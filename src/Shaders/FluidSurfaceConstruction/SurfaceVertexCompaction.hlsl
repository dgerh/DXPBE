#include "SurfaceVertexCompactionRootSig.hlsl"
#include "../constants.h"

// Inputs
// SRV for surface vertices buffer
StructuredBuffer<int> surfaceVertices : register(t0);
// SRV for the dispatch arguments buffer
StructuredBuffer<int3> surfaceBlocksDispatch : register(t1);

// Outputs (UAVs)
RWStructuredBuffer<int> surfaceVertexIndices : register(u0);
RWStructuredBuffer<int3> surfaceVertDensityDispatch : register(u1);

/*
    Similar to SurfaceBlockDetection, this is effectively a stream compaction step. Again, instead of following the paper directly,
    which uses many atomic operations, we'll use wave intrinsics to reduce the number of atomics to one-per-wave.

    This compute pass launches a thread per potential surface vertex. Each workgroup is sized to match a single surface block, so the total threads are (CELLS_PER_BLOCK_EDGE + 1)^3 * surfaceBlocksDispatch[0].x.
    Note that the threads in a single workgroup don't necessarily all map to the same surface block.
*/
[numthreads(SURFACE_VERTEX_COMPACTION_THREADS_X, 1, 1)]
void main(uint3 globalThreadId : SV_DispatchThreadID) {
    if (globalThreadId.x >= surfaceBlocksDispatch[0].x * (CELLS_PER_BLOCK_EDGE + 1) * (CELLS_PER_BLOCK_EDGE + 1) * (CELLS_PER_BLOCK_EDGE + 1)) {
        return;
    }

    bool isSurfaceVertex = surfaceVertices[globalThreadId.x] > 0;  
    int localSurfaceVertexIndexInWave = WavePrefixCountBits(isSurfaceVertex);
    int surfaceVertexWaveCount = WaveActiveCountBits(isSurfaceVertex);
    int surfaceVertexGlobalStartIdx;

    if (WaveIsFirstLane()) {
        InterlockedAdd(surfaceVertDensityDispatch[0].x, surfaceVertexWaveCount, surfaceVertexGlobalStartIdx);
        // No synchronziation is necessary, because we only care about the value within a wave.
    }
    surfaceVertexGlobalStartIdx = WaveReadLaneFirst(surfaceVertexGlobalStartIdx); // (its faster to use an intrinsic here than to write to a shared variable and sync)

    if (isSurfaceVertex) {
        int surfaceVertexGlobalIdx = surfaceVertexGlobalStartIdx + localSurfaceVertexIndexInWave;
        surfaceVertexIndices[surfaceVertexGlobalIdx] = globalThreadId.x;
    }
}