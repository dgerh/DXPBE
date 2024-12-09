#include "BilevelUniformGridRootSig.hlsl"
#include "utils.hlsl"
#include "../constants.h"

// SRV for positions buffer (input buffer)
StructuredBuffer<float4> positionsBuffer : register(t0);
StructuredBuffer<int> materialsBuffer : register(t1);

// UAV for the bilevel uniform grid (output buffers)
RWStructuredBuffer<int> cellParticleCounts : register(u0);
RWStructuredBuffer<int> cellParticleIndices : register(u1);
RWStructuredBuffer<int> blocks : register(u2);

ConstantBuffer<BilevelUniformGridConstants> cb : register(b0);

int3 getCellIndex(float3 particlePosition) {
    int cellIdxX = floor((particlePosition.x - cb.minBounds.x) / cb.resolution);
    int cellIdxY = floor((particlePosition.y - cb.minBounds.y) / cb.resolution);
    int cellIdxZ = floor((particlePosition.z - cb.minBounds.z) / cb.resolution);

    return int3(cellIdxX, cellIdxY, cellIdxZ);
}

// NOTE: if this compute shader changes to 3D, the logic also needs to change to get and use the particle index correctly.
[numthreads(BILEVEL_UNIFORM_GRID_THREADS_X, 1, 1)]
void main(uint3 globalThreadId : SV_DispatchThreadID) {
    if (globalThreadId.x >= cb.numParticles) {
        return;
    }

    // Check that the particle is water, return otherwise
	if (materialsBuffer[globalThreadId.x] != 0) {
		return;
	}

    float3 position = positionsBuffer[globalThreadId.x].xyz;
    int3 cellIndices = getCellIndex(position);
    int cellIndex1D = to1D(cellIndices, cb.dimensions);
    int3 blockIndices = cellIndices / CELLS_PER_BLOCK_EDGE;
    int3 localCellIndices = cellIndices - (blockIndices * CELLS_PER_BLOCK_EDGE); // could be done with modulo, but this is faster since we already have blockIndices

    // (±1 or 0, ±1 or 0, ±1 or 0)
    // 1 indicates cell is on a positive edge of a block, -1 indicates cell is on a negative edge, 0 indicates cell is not on an edge
    // This should work for both even and odd CELLS_PER_BLOCK_EDGE
    float halfCellsPerBlockEdgeMinusOne = ((CELLS_PER_BLOCK_EDGE - 1) / 2.0);
    int3 edge = int3(trunc((localCellIndices - halfCellsPerBlockEdgeMinusOne) / halfCellsPerBlockEdgeMinusOne));

    // Add this particle to the cell
    int particleIndexInCell;
    InterlockedAdd(cellParticleCounts[cellIndex1D], 1, particleIndexInCell);
    if (particleIndexInCell < MAX_PARTICLES_PER_CELL) {
        cellParticleIndices[cellIndex1D * MAX_PARTICLES_PER_CELL + particleIndexInCell] = globalThreadId.x;
    }

    // Do this only once per cell, when the first particle is added to the cell
    if (particleIndexInCell == 0) {
        // Increment the number of non-empty cells in the block, and any blocks for which this cell borders on (which can be a maximum of 8)
        // `edge` is carefully calculated so that these nested loops will exactly iterate over each abutting neighbor block.
        int3 gridBlockDimensions = cb.dimensions / CELLS_PER_BLOCK_EDGE;
        int3 globalNeighborBlockIndex3d = clamp(blockIndices + edge, int3(0, 0, 0), gridBlockDimensions - 1);
        int3 minSearchBounds = min(blockIndices, globalNeighborBlockIndex3d);
        int3 maxSearchBounds = max(blockIndices, globalNeighborBlockIndex3d);

        for (int z = minSearchBounds.z; z <= maxSearchBounds.z; ++z) {
            for (int y = minSearchBounds.y; y <= maxSearchBounds.y; ++y) {
                for (int x = minSearchBounds.x; x <= maxSearchBounds.x; ++x) {
                    int3 neighborBlockIndices = int3(x, y, z);
                    int neighborBlockIndex1D = to1D(neighborBlockIndices, gridBlockDimensions);

                    // TODO: test if its faster to atomic add first to a shared memory variable, then add to the buffer at the end
                    InterlockedAdd(blocks[neighborBlockIndex1D], 1);
                }
            }
        }
    }
}