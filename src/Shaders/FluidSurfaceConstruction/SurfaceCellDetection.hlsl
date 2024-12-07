#include "SurfaceCellDetectionRootSig.hlsl"
#include "../constants.h"
#include "utils.hlsl"

struct Uniforms {
    int3 gridCellDimensions;
};

// Inputs
// Constant buffer (root constant)
ConstantBuffer<Uniforms> cb : register(b0);
// SRV for surface block indices
StructuredBuffer<int> surfaceBlockIndices : register(t0);
// SRV for the surface cells
StructuredBuffer<Cell> cells : register(t1);
// SRV for the surface block dispatch
StructuredBuffer<int3> surfaceBlockDispatch : register(t2);

// Outputs
RWStructuredBuffer<int> surfaceVertices : register(u0);

RWStructuredBuffer<int3> surfaceHalfBlockDispatch : register(u1); // piggy-backing off this pass to set up an indirect dispatch for the mesh shading step

// At a typical value of FILLED_BLOCK = 216, (4 + 2)^3, this is well within the limits of shared memory. 
groupshared int cellParticleCounts[FILLED_BLOCK];

// Since surface cells share vertices, keep track of whether the verts are surface verts in shared memory first,
// then write them to the output buffer. This avoids excess global memory writes.
groupshared int s_surfaceVertices[(CELLS_PER_BLOCK_EDGE + 1) * (CELLS_PER_BLOCK_EDGE + 1) * (CELLS_PER_BLOCK_EDGE + 1)];

// Set the 8 vertices of a surface cell cube into shared memory
void setSharedMemSurfaceVerts(int3 baseIndex3d, int3 dimensions, int value) {
    s_surfaceVertices[to1D(baseIndex3d + int3(0, 0, 0), dimensions)] = value;
    s_surfaceVertices[to1D(baseIndex3d + int3(0, 0, 1), dimensions)] = value;
    s_surfaceVertices[to1D(baseIndex3d + int3(0, 1, 0), dimensions)] = value;
    s_surfaceVertices[to1D(baseIndex3d + int3(0, 1, 1), dimensions)] = value;
    s_surfaceVertices[to1D(baseIndex3d + int3(1, 0, 0), dimensions)] = value;
    s_surfaceVertices[to1D(baseIndex3d + int3(1, 0, 1), dimensions)] = value;
    s_surfaceVertices[to1D(baseIndex3d + int3(1, 1, 0), dimensions)] = value;
    s_surfaceVertices[to1D(baseIndex3d + int3(1, 1, 1), dimensions)] = value;
}

// Set the 8 vertices of a surface cell cube into the output buffer
void setGlobalSurfaceVertsToValue(int3 globalIndex3d, int3 globalDims, int value) {
    surfaceVertices[to1D(globalIndex3d + int3(0, 0, 0), globalDims)] = value;
    surfaceVertices[to1D(globalIndex3d + int3(0, 0, 1), globalDims)] = value;
    surfaceVertices[to1D(globalIndex3d + int3(0, 1, 0), globalDims)] = value;
    surfaceVertices[to1D(globalIndex3d + int3(0, 1, 1), globalDims)] = value;
    surfaceVertices[to1D(globalIndex3d + int3(1, 0, 0), globalDims)] = value;
    surfaceVertices[to1D(globalIndex3d + int3(1, 0, 1), globalDims)] = value;
    surfaceVertices[to1D(globalIndex3d + int3(1, 1, 0), globalDims)] = value;
    surfaceVertices[to1D(globalIndex3d + int3(1, 1, 1), globalDims)] = value;
}

// Set the 8 vertices of a surface cell cube into the output buffer, from shared memory.
void setGlobalSurfaceVertsFromSharedMem(int3 globalIndex3d, int3 globalDims, int3 sharedIndex3d, int3 sharedDims) {
    surfaceVertices[to1D(globalIndex3d + int3(0, 0, 0), globalDims)] = s_surfaceVertices[to1D(sharedIndex3d + int3(0, 0, 0), sharedDims)];
    surfaceVertices[to1D(globalIndex3d + int3(0, 0, 1), globalDims)] = s_surfaceVertices[to1D(sharedIndex3d + int3(0, 0, 1), sharedDims)];
    surfaceVertices[to1D(globalIndex3d + int3(0, 1, 0), globalDims)] = s_surfaceVertices[to1D(sharedIndex3d + int3(0, 1, 0), sharedDims)];
    surfaceVertices[to1D(globalIndex3d + int3(0, 1, 1), globalDims)] = s_surfaceVertices[to1D(sharedIndex3d + int3(0, 1, 1), sharedDims)];
    surfaceVertices[to1D(globalIndex3d + int3(1, 0, 0), globalDims)] = s_surfaceVertices[to1D(sharedIndex3d + int3(1, 0, 0), sharedDims)];
    surfaceVertices[to1D(globalIndex3d + int3(1, 0, 1), globalDims)] = s_surfaceVertices[to1D(sharedIndex3d + int3(1, 0, 1), sharedDims)];
    surfaceVertices[to1D(globalIndex3d + int3(1, 1, 0), globalDims)] = s_surfaceVertices[to1D(sharedIndex3d + int3(1, 1, 0), sharedDims)];
    surfaceVertices[to1D(globalIndex3d + int3(1, 1, 1), globalDims)] = s_surfaceVertices[to1D(sharedIndex3d + int3(1, 1, 1), sharedDims)];
}

void setVertSharedMemory(int index, int value) {
    s_surfaceVertices[index] = value;
}

void setVertGlobalMemory(int index, int value) {
    surfaceVertices[index] = value;
}

// NOTE: the logic in this shader RELIES on the number of threads per workgroup equaling the number of cells in a single block. 
// (it can be changed not to, but doing so avoids the use of a modulo operation)
[numthreads(SURFACE_CELL_DETECTION_THREADS, SURFACE_CELL_DETECTION_THREADS, SURFACE_CELL_DETECTION_THREADS)]
void main(uint3 localThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID) {
    // The built-in global thread ID indexes in a different order than we want for our grid, so calculate it ourselves as such:
    uint globalThreadId = to1D(localThreadId, int3(SURFACE_CELL_DETECTION_THREADS, SURFACE_CELL_DETECTION_THREADS, SURFACE_CELL_DETECTION_THREADS))
                            + (groupId.x * SURFACE_CELL_DETECTION_THREADS * SURFACE_CELL_DETECTION_THREADS * SURFACE_CELL_DETECTION_THREADS);

    if (globalThreadId >= surfaceBlockDispatch[0].x * CELLS_PER_BLOCK) {
        return;
    }

    // Piggy-backing off this pass to set up an indirect dispatch for the mesh shading step
    if (globalThreadId == 0) {
        surfaceHalfBlockDispatch[0] = int3(surfaceBlockDispatch[0].x * 2, 1, 1);
    }

    // First order of business: we launched a thread for each cell within surface blocks. We need to figure out the global index for this thread's cell. 
    int surfaceBlockIdx1d = surfaceBlockIndices[globalThreadId / CELLS_PER_BLOCK];
    // By aligning the number of threads per workgroup with the number of threads in a block, we can use the local thread ID as a proxy for the local cell index.
    int3 surfaceBlockIdx3d = to3D(surfaceBlockIdx1d, cb.gridCellDimensions / CELLS_PER_BLOCK_EDGE);
    int3 localCellIndex3d = int3(localThreadId.x, localThreadId.y, localThreadId.z);
    int3 globalCellIndex3d = surfaceBlockIdx3d * CELLS_PER_BLOCK_EDGE + localCellIndex3d;

    // Prefetch and store the particle count for this cell. Since shared memory will store (N + 2)^3 cells, not just N^3 cells, in order for all cells to be contiguous,
    // we need to offset the surface block cells themselves by 1 to leave room for the extra-surface cells.
    // (Skip the cells on edges, we'll get them later)
    int offsetLocalCellIdx1d = to1D(localCellIndex3d + int3(1, 1, 1), CELLS_PER_BLOCK_EDGE + 2);
    if (all(localCellIndex3d < (CELLS_PER_BLOCK_EDGE - 1) * int3(1, 1, 1)) && all(localCellIndex3d > int3(0, 0, 0))) {
        cellParticleCounts[offsetLocalCellIdx1d] = cells[globalThreadId].particleCount;
    }

    // We also need the particle counts for the cells surrounding the surface block. Let each edge cell in the surface block
    // fetch its surrounding extra-surface cells (and itself).
    // This follows the same strategy as the bilevel uniform grid shader.
    float halfCellsPerBlockEdgeMinusOne = ((CELLS_PER_BLOCK_EDGE - 1) / 2.0);
    int3 edge = int3(trunc((localCellIndex3d - halfCellsPerBlockEdgeMinusOne) / halfCellsPerBlockEdgeMinusOne));
    // By converting neighbors to global-space here, we can clamp to the grid bounds and avoid 
    // out-of-bounds checks in every loop iteration.
    int3 globalNeighborCells = clamp(globalCellIndex3d + edge, int3(0, 0, 0), cb.gridCellDimensions - 1);
    int3 minSearchBounds = min(globalNeighborCells, globalCellIndex3d);
    int3 maxSearchBounds = max(globalNeighborCells, globalCellIndex3d);

    int3 globalNeighborCellOrigin = surfaceBlockIdx3d * CELLS_PER_BLOCK_EDGE - int3(1, 1, 1);
    for (int z = minSearchBounds.z; z <= maxSearchBounds.z; z++) {
        for (int y = minSearchBounds.y; y <= maxSearchBounds.y; y++) {
            for (int x = minSearchBounds.x; x <= maxSearchBounds.x; x++) {
                int3 globalNeighborCellIndex3d = int3(x, y, z);
                int globalNeighborCellIndex1d = to1D(globalNeighborCellIndex3d, cb.gridCellDimensions);
                
                int3 localNeighborCellIndex3d = globalNeighborCellIndex3d - globalNeighborCellOrigin;
                int localNeighborCellIndex1d = to1D(localNeighborCellIndex3d, CELLS_PER_BLOCK_EDGE + 2);
                
                cellParticleCounts[localNeighborCellIndex1d] = cells[globalNeighborCellIndex1d].particleCount;
            }
        }
    }

    // Now that all the memory we'll need is pulled into shared, sync it, and then process it to detect surface cells.
    GroupMemoryBarrierWithGroupSync();

    // Similar trick as before to get the search bounds, but this time we're iterating over the full range of block cells (within grid bounds).
    minSearchBounds = clamp(globalCellIndex3d - int3(1, 1, 1), int3(0, 0, 0), cb.gridCellDimensions - 1);
    maxSearchBounds = clamp(globalCellIndex3d + int3(1, 1, 1), int3(0, 0, 0), cb.gridCellDimensions - 1);

    // A cell is NOT a surface cell if either:
    // 1. All of its neighbors and itself are empty
    // 2. All of its neighbors and itself are filled
    // It CAN be a surface cell EVEN if it has no particles itself.
    // As we iterate over all neighbors, if any one of them is different from the previous ones, the cell is a surface cell.
    // To track this, initialize firstCellEmpty to the state of the current cell, from shared memory.
    bool firstCellEmpty = (cellParticleCounts[offsetLocalCellIdx1d] == 0);
    bool isSurfaceCell = false;
    for (int z = minSearchBounds.z; z <= maxSearchBounds.z; z++) {
        for (int y = minSearchBounds.y; y <= maxSearchBounds.y; y++) {
            for (int x = minSearchBounds.x; x <= maxSearchBounds.x; x++) {
                int3 globalNeighborCellIndex3d = int3(x, y, z);

                int3 localNeighborCellIndex3d = globalNeighborCellIndex3d - globalNeighborCellOrigin;
                int localNeighborCellIndex1d = to1D(localNeighborCellIndex3d, CELLS_PER_BLOCK_EDGE + 2);

                bool isCellEmpty = (cellParticleCounts[localNeighborCellIndex1d] == 0);
                if (isCellEmpty != firstCellEmpty) {
                    isSurfaceCell = true;
                    break;
                };
            }
            if (isSurfaceCell) {
                break;
            }
        }
        if (isSurfaceCell) {
            break;
        }
    }

    // Initialize the shared memory to all 0s, then only write to it if isSurfaceCell.
    // We write to shared memory first because cells can share vertices, and may not agree on what's a surface vertex,
    // so this saves on redundant global writes/overwrites (and the need for atomics).
    setSharedMemSurfaceVerts(localCellIndex3d, (CELLS_PER_BLOCK_EDGE + 1) * int3(1, 1, 1), 0);
    GroupMemoryBarrierWithGroupSync();

    if (isSurfaceCell) {
        setSharedMemSurfaceVerts(localCellIndex3d, (CELLS_PER_BLOCK_EDGE + 1) * int3(1, 1, 1), 1);
    }
    GroupMemoryBarrierWithGroupSync();

    // Set surface vertices buffer
    if (isSurfaceCell) {
        setGlobalSurfaceVertsFromSharedMem(globalCellIndex3d, (cb.gridCellDimensions + 1), localCellIndex3d, (CELLS_PER_BLOCK_EDGE + 1));
    }
}