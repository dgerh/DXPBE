#include "SurfaceVertexDensityRootSig.hlsl"
#include "../constants.h"
#include "utils.hlsl"

// Inputs
// SRV for positions buffer (input buffer)
StructuredBuffer<float3> positionsBuffer : register(t0);
// SRV for the surface cells
StructuredBuffer<Cell> cells : register(t1);
// SRV for the surface vertex indices
StructuredBuffer<int> surfaceVertexIndices : register(t2);
// Root SRV for dispatch params for this pass
StructuredBuffer<int3> surfaceVertDensityDispatch : register(t3);

// Root constants
ConstantBuffer<BilevelUniformGridConstants> cb : register(b0);

// Outputs
// Root UAV for the surface block dispatch (SOLELY to reset it without an extra compute pass)
RWStructuredBuffer<int3> surfaceBlockDispatch : register(u0);
// UAV for the surface vertex densities
RWStructuredBuffer<float> surfaceVertexDensities : register(u1);

float P(float d, float h)
{
    if (d >= 0.0 && d < h) {
        float kernelNorm = 315.0 / (64.0 * PI * pow(h, 9.0));
        return max(0.0, kernelNorm * cubic(h * h - d * d));
    } else {
        return 0.0;
    }
}

float isotropicKernel(float3 r, float h)
{
    r *= KERNEL_SCALE;
    h *= KERNEL_SCALE;
    float d = length(r);
    return P(d / h, h) / cubic(h);
}

// Unfortunately, at this point, threads no longer correspond to spatially-collocated geometry. A single workgroup may span multiple grid blocks.
// This means that we can't take advantage of shared memory to reduce global memory access. It might be worth trying a different approach where 
// workgroups DO correspond to surface blocks, each thread loads the cell data into shared memory, and then those that arent surface threads can retire. (With perhaps an extra index lookup buffer
// to make sure the non-retired threads are contiguous.) It's a tradeoff between extra threads for non-surface verts and extra global memory access.
[numthreads(SURFACE_VERTEX_DENSITY_THREADS_X, 1, 1)]
void main( uint3 globalThreadId : SV_DispatchThreadID ) {
    if (globalThreadId.x >= surfaceVertDensityDispatch[0].x) {
        return;
    }

    // Piggy back off this pass to reset the surface block dispatch buffer for the next frame.
    if (globalThreadId.x == 0) {
        surfaceBlockDispatch[0].x = 0;
    }

    int globalSurfaceVertIndex1d = surfaceVertexIndices[globalThreadId.x];
    int3 globalSurfaceVertIndex3d = to3D(globalSurfaceVertIndex1d, (cb.dimensions + int3(1, 1, 1)));

    float3 vertPos = cb.minBounds + float3(globalSurfaceVertIndex3d) * cb.resolution;
    float totalDensity = 0.0f;

    // In the paper, rather than a static (1, 1, 1) search radius, theirs is variable. That said, they do default it to 1 in their repo, and say as much in the paper. 
    // Anything larger than 1 is honestly prohibitively expensive for real time anyway.
    int3 minLoopBounds = max(globalSurfaceVertIndex3d - int3(1, 1, 1), int3(0, 0, 0));
    int3 maxLoopBounds = min(globalSurfaceVertIndex3d, cb.dimensions - int3(1, 1, 1)); // Note, this is NOT a mistake. Not supposed to add 1 here. Each vertex has at most 8 abutting cells.

    // And now we're iterating over the (maximum) 8 cells that abut the vertex.
    for (int z = minLoopBounds.z; z <= maxLoopBounds.z; z++) {
        for (int y = minLoopBounds.y; y <= maxLoopBounds.y; y++) {
            for (int x = minLoopBounds.x; x <= maxLoopBounds.x; x++) {
                int3 neighborCellIdx3d = int3(x, y, z);
                int neighborCellIdx1d = to1D(neighborCellIdx3d, cb.dimensions);

                int particleCount = cells[neighborCellIdx1d].particleCount;
                for (int i = 0; i < particleCount; i++) {
                    int particleIdx = cells[neighborCellIdx1d].particleIndices[i];
                    float3 particlePos = positionsBuffer[particleIdx];
                    float3 r = vertPos - particlePos;
                    totalDensity += isotropicKernel(r, cb.resolution); // (In the paper, they use kernel radius here, which is defaulted to 0.99 * cell resolution)
                }
            }
        }
    }

    // Densities aren't compresed but the only populated entries correspond to verts of surface blocks (of which not all are surface verts).
    surfaceVertexDensities[globalSurfaceVertIndex1d] = totalDensity;
}