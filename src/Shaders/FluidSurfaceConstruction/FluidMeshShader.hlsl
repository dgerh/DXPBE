#include "FluidMeshRootSig.hlsl"
#include "../constants.h"
#include "utils.hlsl"
#include "MarchingCubesTables.hlsl"

// Inputs
// SRV for the surface block indices
StructuredBuffer<int> surfaceBlockIndices : register(t0);
// SRV for vertex densities
StructuredBuffer<float> vertexDensities : register(t1);
// SRV for vertex normals
StructuredBuffer<float3> vertexNormals : register(t2);
// SRV for dispatch parameters
StructuredBuffer<int3> surfaceHalfBlockDispatch : register(t3);
// Root constants
ConstantBuffer<MeshShadingConstants> cb : register(b0);
// UAV
RWStructuredBuffer<int3> surfaceVertDensityDispatch : register(u0); // purely for resetting the buffer

// Shared memory: 170 edges in a 4x4x2 halfblock, total of (1 + 3 + 2 + 4) * 4 (Bytes/word) = 40 bytes per vertex.
// At most 170 vertices (one per edge) can be constructed -> 170 * 36 = 6.8KB shared memory per workgroup. This is well within size limits.
// Note: due to the way this shared memory is set and accessed, there's no need to initialize it to zero. Only set entries are accessed.

// Given an edge index, this array gives the corresponding vertex index in the vertex output buffer
groupshared int outputVertexIndices[MAX_VERTICES];
// Rest are vertex attributes (indexed by outputVertexIndices)
groupshared float3 vertexWorldPositions[MAX_VERTICES];
groupshared float3 vertexNormalsShared[MAX_VERTICES];
groupshared float4 vertexClipPositions[MAX_VERTICES];
groupshared float vertexMeshletIndices[MAX_VERTICES];

// Define the payload structure
struct VertexOutput
{
    float4 clipPos : SV_POSITION;
    float3 normal : NORMAL0;
    float3 worldPos : POSITION1;
    int meshletIndex : COLOR0;
};

// Get the global grid vertex index of both endpoints from the edge index in the block
// 
// blockEdge:         [0, 169]
// halfBlockIndex: [0, 1]
// edgeDims represent the dimensions of the edges in a halfblock in each axis
static const int3 edgeDims[3] = { int3(4, 5, 3), int3(5, 4, 3), int3(5, 5, 2) };
int3 getGlobalVerticesForEdge(int3 blockIndices, int blockEdge, int halfBlockIndex)[2] {
    // For a 4x4x2 halfblock, there are 60 edges in the x and y axis and 50 in the z axis, so
    // a divisor / modulus of 60 can be used to find the axis / edge index within the axis.
    int axis = blockEdge / 60;
    int edgeIdxInAxis1d = blockEdge % 60;

    // Since each edge stems from a vertex, we can exploit that 1:1 mapping and treat the edge index
    // as a local vertex index.
    int3 localVertexIndex3D = to3D(edgeIdxInAxis1d, edgeDims[axis]);
    
    int3 globalVertexIndex3d = (blockIndices * CELLS_PER_BLOCK_EDGE) + localVertexIndex3D;
    globalVertexIndex3d.z += halfBlockIndex * 2;

    int3 offset = int3(0, 0, 0);
    offset[axis] = 1;

    int3 vertices[2];
    vertices[0] = globalVertexIndex3d;
    vertices[1] = globalVertexIndex3d + offset;
    
    return vertices;
}

float3 getVertexNormals(int3 vertexIndices[2])[2] {
    float3 normal0 = vertexNormals[to1D(vertexIndices[0], (cb.dimensions + int3(1, 1, 1)))];
    float3 normal1 = vertexNormals[to1D(vertexIndices[1], (cb.dimensions + int3(1, 1, 1)))];

    float3 normals[2] = {normal0, normal1};
    return normals;
}

// In marching cubes, we place new vertices along edges according to the density values at the endpoints, using linear interpolation.
float interpolateDensity(float d0, float d1) {
    // TODO: this seems a little overcomplicated. Can it be simplified?
    if (abs(d0 - ISOVALUE) < EPSILON && abs(d1 - ISOVALUE) < EPSILON) {
        return 0.5;
    }

    if (abs(d0 - d1) > EPSILON) {
        float t = clamp((ISOVALUE - d0) / (d1 - d0), 0.0, 1.0);
        return t;
    }

    return d0 < ISOVALUE ? 1.0 : 0.0;
}

// In marching cubes, we draw triangles in a cube depending on which of the 8 vertices are above the isovalue. With 8 vertices in a cube,
// there are 2^8 = 256 possible cases. We can compute which case we're in as a bitfield by comparing the density values at each vertex to the isovalue and bitshifting.
int computeMarchingCubesCase(int3 globalCellIndices) {
    // TODO: this is a LOT of global memory reads... (and redundant ones at that)
    int mcCase = 0;
    int3 globalVertDims = (cb.dimensions + int3(1, 1, 1));
    
    mcCase += (vertexDensities[to1D(globalCellIndices + int3(0, 0, 0), globalVertDims)] > ISOVALUE) << 0;
    mcCase += (vertexDensities[to1D(globalCellIndices + int3(1, 0, 0), globalVertDims)] > ISOVALUE) << 1;
    mcCase += (vertexDensities[to1D(globalCellIndices + int3(0, 0, 1), globalVertDims)] > ISOVALUE) << 2;
    mcCase += (vertexDensities[to1D(globalCellIndices + int3(1, 0, 1), globalVertDims)] > ISOVALUE) << 3;
    mcCase += (vertexDensities[to1D(globalCellIndices + int3(0, 1, 0), globalVertDims)] > ISOVALUE) << 4;
    mcCase += (vertexDensities[to1D(globalCellIndices + int3(1, 1, 0), globalVertDims)] > ISOVALUE) << 5;
    mcCase += (vertexDensities[to1D(globalCellIndices + int3(0, 1, 1), globalVertDims)] > ISOVALUE) << 6;
    mcCase += (vertexDensities[to1D(globalCellIndices + int3(1, 1, 1), globalVertDims)] > ISOVALUE) << 7;

    return mcCase;
}

// Takes in an edge index within a cell [0, 11] and returns the edge index within a halfblock [0, 169]
// In `getGlobalVerticesForEdge`, we implicitly treated ascending block-edge indices as being all along one axis first, then the next, then the next.
// Moreover, in setting up the Marching Cubes tables, we implicitly assigned an order to local cell-edge indices. 
// Both of these implicit assignments need to be considered to map from a cell edge to a block edge.
//
// See MarchingCubesTables.hlsl for local edge orderings. The offsets map a local edge to a block edge.
static const int edgeOffsets[12] = {0, 4, 20, 24, 60, 61, 80, 81, 120, 121, 125, 126};
int cellEdgeToBlockEdge(int localCellIdx1d, int localEdgeIdx, int halfBlockIndex) {
    int axis = localEdgeIdx / 4;
    localCellIdx1d -= halfBlockIndex * CELLS_PER_HALFBLOCK;
    int3 localCellIdx3d = to3D(localCellIdx1d, CELLS_PER_BLOCK_EDGE * int3(1, 1, 1));

    // Put back into 1D, but using the dimensions of the edges along a given axis
    int dimensionOffset = to1D(localCellIdx3d, edgeDims[axis]);

    return edgeOffsets[localEdgeIdx] + dimensionOffset;    
}

[outputtopology("triangle")]
// Each workgroup represents half a block of cells. (To appease mesh shading limits on output number of prims/verts)
// Each thread will represent a single cell (not necessarily a surface cell), but process multiple edges.
[numthreads(CELLS_PER_HALFBLOCK, 1, 1)]
void main(
    uint3 globalThreadId : SV_DispatchThreadID,
    uint3 localThreadId : SV_GroupThreadID, 
    uint3 workgroupId : SV_GroupID,
    out vertices VertexOutput verts[MAX_VERTICES],
    out indices uint3 triangles[MAX_PRIMITIVES])
{
    if (globalThreadId.x >= surfaceHalfBlockDispatch[0].x * CELLS_PER_HALFBLOCK) return;

    // Piggy back to reset the surfaceVertDensityDispatch buffer
    if (globalThreadId.x == 0) {
        surfaceVertDensityDispatch[0].x = 0;
    }

    int blockIdx1d = surfaceBlockIndices[globalThreadId.x / CELLS_PER_BLOCK];
    int3 blockIdx3d = to3D(blockIdx1d, (cb.dimensions / CELLS_PER_BLOCK_EDGE));
    
    int localCellIdx1d = globalThreadId.x % CELLS_PER_BLOCK;
    int halfBlockIndex = (localCellIdx1d < (CELLS_PER_BLOCK / 2)) ? 0 : 1;

    // Each thread processes 170 / workgroup_size edges (170 is the number of edges in a 4x4x2 half block)
    int vertexCount = 0;

    for (int i = 0; i < divRoundUp(EDGES_PER_HALFBLOCK, CELLS_PER_HALFBLOCK); ++i) {
        int edgeIdx = i * CELLS_PER_HALFBLOCK + localThreadId.x;
        bool needVertex = false;
        float density0;
        float density1;
        int3 vertexIndices[2];

        // Since we're not dealing with even multiples, some threads will have out of bounds edge indices to deal with.
        // We still need to increment the vertexCounts for these threads though, so we can't just continue early, yet.
        if (edgeIdx < EDGES_PER_HALFBLOCK) { 
            vertexIndices = getGlobalVerticesForEdge(blockIdx3d, edgeIdx, halfBlockIndex);
            density0 = vertexDensities[to1D(vertexIndices[0], (cb.dimensions + int3(1, 1, 1)))];
            density1 = vertexDensities[to1D(vertexIndices[1], (cb.dimensions + int3(1, 1, 1)))];

            needVertex = ((density0 > ISOVALUE) ^ (density1 > ISOVALUE));
        }

        // To get an offset into shared memory, each thread can get the partial sum number of verts
        // from lower-index threads in this wave, using WavePrefixCountBits. (This relies on wave size = group size = 32).
        int outputVertexIndex = vertexCount + WavePrefixCountBits(needVertex);
        // (Each thread is keeping track of an individual copy of vertexCount, but staying in sync via wave intrinsics) 
        vertexCount += WaveActiveCountBits(needVertex);

        if (!needVertex) continue;
        
        float3 vertexNormals[2] = getVertexNormals(vertexIndices);
        
        float t = interpolateDensity(density0, density1);
        float3 vertPosWorld = cb.minBounds + cb.resolution * lerp(float3(vertexIndices[0]), float3(vertexIndices[1]), t);
        float4 vertPosClip = mul(cb.viewProj, float4(vertPosWorld, 1.0));
        float3 vertNormal = -normalize(lerp(vertexNormals[0], vertexNormals[1], t)); // Paper negates normals, not sure why!

        // Store the index of the output vertex in shared memory
        // In next step, each thread acts as a cell and will read several vertex indices, so we need them all in shared memory.
        outputVertexIndices[edgeIdx] = outputVertexIndex;
        // Then store all the vertex attributes at this outputVertexIndex
        // (Note, HLSL has a mesh shader restriction: you have to tell it the total output verts + prims there are BEFORE you write to them)
        // (For that reason, mostly, we write the vertex attributes to shared memory first; otherwise we could just write them output right now, before we know the total count).
        vertexWorldPositions[outputVertexIndex] = vertPosWorld;
        vertexClipPositions[outputVertexIndex] = vertPosClip;
        vertexNormalsShared[outputVertexIndex] = vertNormal;
        vertexMeshletIndices[outputVertexIndex] = workgroupId.x;
    }

    GroupMemoryBarrierWithGroupSync();

    // From here on out, every thread acts as a single cell (TODO: this is where I want to try optimizing; return early for non-surface cells)
    int3 localCellIdx3d = to3D(localCellIdx1d, CELLS_PER_BLOCK_EDGE * int3(1, 1, 1)); // TODO: can we avoid this conversion with 3D dispatch?
    int3 globalCellIdx3d = blockIdx3d * CELLS_PER_BLOCK_EDGE + localCellIdx3d;

    int mcCase = computeMarchingCubesCase(globalCellIdx3d);
    int numTris = triangleCounts[mcCase];
    int triOffset = WavePrefixSum(numTris);
    int totalTris = WaveActiveSum(numTris);

    // We finally have all the info we need to tell the mesh shader how many vertices and primitives we're outputting.
    SetMeshOutputCounts(vertexCount, totalTris);

    // Every surface block has surface vertices, but since each workgroup represents a *half*-blocks,
    // it's possible for a workgroup's half block to have no surface vertices. 
    // Note: paper does this early-return before MC case computation (which is great because it's cheaper). Unfortunately I believe it's undefined behavior,
    // because all threads have to set mesh output counts.
    if (vertexCount == 0) return;

    // Now we do the actual marching cubes / outputting of vertices and tris
    int triTable[12] = triangleTable[mcCase];
    for (int t = 0; t < numTris; ++t) {
        int triIndices[3];

        [unroll]
        for (int v = 0; v < 3; ++v) {
            int cellEdgeIdx = triTable[t * 3 + v]; // edge index within a cell [0-11]
            int blockEdgeIdx = cellEdgeToBlockEdge(localCellIdx1d, cellEdgeIdx, halfBlockIndex); // edge within a block (0-169)
            int outputVertexIndex = outputVertexIndices[blockEdgeIdx];

            triIndices[v] = outputVertexIndex;

            // Write the vertex attributes to the output buffer
            verts[outputVertexIndex].clipPos = vertexClipPositions[outputVertexIndex];
            verts[outputVertexIndex].normal = vertexNormalsShared[outputVertexIndex];
            verts[outputVertexIndex].worldPos = vertexWorldPositions[outputVertexIndex];
            verts[outputVertexIndex].meshletIndex = blockIdx1d;
        }
        
        // Write the triangle to the output buffer
        triangles[triOffset + t] = uint3(triIndices[0], triIndices[1], triIndices[2]);
    }
}
 