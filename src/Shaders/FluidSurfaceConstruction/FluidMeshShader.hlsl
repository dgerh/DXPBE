#include "FluidMeshRootSig.hlsl"
#include "../constants.h"
#include "utils.hlsl"
#include "MarchingCubesTables.hlsl"

// Inputs
// SRV for the surface block indices
StructuredBuffer<uint> surfaceBlockIndices : register(t0);
// SRV for vertex densities
StructuredBuffer<float> vertexDensities : register(t1);
// SRV for vertex normals
StructuredBuffer<float2> vertexNormals : register(t2);
// SRV for dispatch parameters
StructuredBuffer<uint3> surfaceHalfBlockDispatch : register(t3);
// Root constants
ConstantBuffer<MeshShadingConstants> cb : register(b0);
// UAV
RWStructuredBuffer<uint3> surfaceVertDensityDispatch : register(u0); // purely for resetting the buffer

// Shared memory: 170 edges in a 4x4x2 halfblock, total of (1 + 3 + 2 + 4) * 4 (Bytes/word) = 40 bytes per vertex.
// At most 170 vertices (one per edge) can be constructed -> 170 * 36 = 6.8KB shared memory per workgroup. This is well within size limits.
// Note: due to the way this shared memory is set and accessed, there's no need to initialize it to zero. Only set entries are accessed.

// Given an edge index, this array gives the corresponding vertex index in the vertex output buffer
groupshared uint outputVertexIndices[MAX_VERTICES];
// Rest are vertex attributes (indexed by outputVertexIndices)
groupshared float3 vertexWorldPositions[MAX_VERTICES];
groupshared float2 vertexNormalsShared[MAX_VERTICES]; // z normal component can be inferred in frag shader
groupshared float4 vertexClipPositions[MAX_VERTICES];

// Define the payload structure
struct VertexOutput
{
    float4 clipPos : SV_POSITION;
    float2 normal : NORMAL;
    float3 worldPos : TEXCOORD0;
};

// Get the global grid vertex index of both endpoints from the edge index in the block
// 
// blockEdge:         [0, 169]
// halfBlockIndex: [0, 1]
// edgeDims represent the dimensions of the edges in a halfblock in each axis
static const float3 edgeDims[3] = { float3(4, 5, 3), float3(5, 4, 3), float3(5, 5, 2) };
uint3 getGlobalVerticesForEdge(float3 blockIndices, uint blockEdge, uint halfBlockIndex)[2] {
    // For a 4x4x2 halfblock, there are 60 edges in the x and y axis and 50 in the z axis, so
    // a divisor / modulus of 60 can be used to find the axis / edge index within the axis.
    uint axis = blockEdge / 60;
    uint edgeIdxInAxis1d = blockEdge % 60;

    // Since each edge stems from a vertex, we can exploit that 1:1 mapping and treat the edge index
    // as a local vertex index.
    float3 localVertexIndex3D = to3D(edgeIdxInAxis1d, edgeDims[axis]);
    
    float3 globalVertexIndex3d = (blockIndices * CELLS_PER_BLOCK_EDGE) + localVertexIndex3D;
    globalVertexIndex3d.z += halfBlockIndex * 2;

    float3 offset = float3(0, 0, 0);
    offset[axis] = 1;

    uint3 vertices[2];
    vertices[0] = globalVertexIndex3d;
    vertices[1] = globalVertexIndex3d + offset;
    
    return vertices;
}

float3 getVertexNormals(uint3 vertexIndices[2])[2] {
    float3 normal0, normal1;
    normal0.xy = vertexNormals[to1D(vertexIndices[0], (cb.dimensions + 1) * uint3(1, 1, 1))];
    normal1.xy = vertexNormals[to1D(vertexIndices[1], (cb.dimensions + 1) * uint3(1, 1, 1))];

    // Need to infer z component of normals
    normal0 = float3(normal0.xy, sqrt(1 - dot(normal0, normal0)));
    normal1 = float3(normal1.xy, sqrt(1 - dot(normal1, normal1)));

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
uint computeMarchingCubesCase(uint3 globalCellIndices) {
    // TODO: this is a LOT of global memory reads... (and duplicate ones at that)
    uint mcCase = 0;
    uint3 globalVertDims = (cb.dimensions + 1) * uint3(1, 1, 1);
    mcCase += (vertexDensities[to1D(globalCellIndices + uint3(0, 0, 0), globalVertDims)] > ISOVALUE) << 0;
    mcCase += (vertexDensities[to1D(globalCellIndices + uint3(1, 0, 0), globalVertDims)] > ISOVALUE) << 1;
    mcCase += (vertexDensities[to1D(globalCellIndices + uint3(1, 0, 1), globalVertDims)] > ISOVALUE) << 2;
    mcCase += (vertexDensities[to1D(globalCellIndices + uint3(0, 0, 1), globalVertDims)] > ISOVALUE) << 3;
    mcCase += (vertexDensities[to1D(globalCellIndices + uint3(0, 1, 0), globalVertDims)] > ISOVALUE) << 4;
    mcCase += (vertexDensities[to1D(globalCellIndices + uint3(1, 1, 0), globalVertDims)] > ISOVALUE) << 5;
    mcCase += (vertexDensities[to1D(globalCellIndices + uint3(1, 1, 1), globalVertDims)] > ISOVALUE) << 6;
    mcCase += (vertexDensities[to1D(globalCellIndices + uint3(0, 1, 1), globalVertDims)] > ISOVALUE) << 7;
    return mcCase;
}

// Takes in an edge index within a cell [0, 11] and returns the edge index within a halfblock [0, 169]
// In `getGlobalVerticesForEdge`, we implicitly treated ascending block-edge indices as being all along one axis first, then the next, then the next.
// Moreover, in setting up the Marching Cubes tables, we implicitly assigned an order to local cell-edge indices. 
// Both of these implicit assignments need to be considered to map from a cell edge to a block edge.
static const int edgeOffsets[12] = {0, 61, 4, 60, 20, 81, 24, 80, 120, 121, 126, 125};
uint cellEdgeToBlockEdge(uint localCellIdx1d, uint localEdgeIdx, uint halfBlockIndex) {
    localCellIdx1d -= halfBlockIndex * CELLS_PER_HALFBLOCK;
    
    uint firstEdge = localCellIdx1d % 16 + (localCellIdx1d / 16) * 20;
    uint3 localCellIdx3d = to3D(localCellIdx1d, CELLS_PER_BLOCK_EDGE);
    uint j = localCellIdx3d.y;
    uint k = localCellIdx3d.z;

    uint val = firstEdge + edgeOffsets[localEdgeIdx];
    if (localEdgeIdx >= 8){
        return val + j + 5 * k;
    } else if(localEdgeIdx % 2 == 1){
        return val + j;
    }
    return val;
}

[outputtopology("triangle")]
// Each workgroup represents half a block of cells. (To appease mesh shading limits on output number of prims/verts)
// Each thread will represent a single cell (not necessarily a surface cell), but process multiple edges.
[numthreads(CELLS_PER_HALFBLOCK, 1, 1)]
void main(
    uint3 globalThreadId : SV_DispatchThreadID,
    uint3 localThreadId : SV_GroupThreadID, 
    out vertices VertexOutput verts[MAX_VERTICES],
    out indices uint3 triangles[MAX_PRIMITIVES])
{
    if (globalThreadId.x > surfaceHalfBlockDispatch[0].x * CELLS_PER_HALFBLOCK) return;

    // Piggy back to reset the surfaceVertDensityDispatch buffer
    if (globalThreadId.x == 0) {
        surfaceVertDensityDispatch[0].x = 0;
    }

    uint blockIdx1d = surfaceBlockIndices[globalThreadId.x / CELLS_PER_BLOCK];
    uint3 blockIdx3d = to3D(blockIdx1d, (cb.dimensions / CELLS_PER_BLOCK_EDGE) * uint3(1, 1, 1));

    uint halfBlockIndex = (localThreadId.x < CELLS_PER_BLOCK / 2) ? 0 : 1;

    // Each thread processes 170 / workgroup_size edges (170 is the number of edges in a 4x4x2 half block)
    uint vertexCount = 0;

    for (int i = 0; i < divRoundUp(EDGES_PER_HALFBLOCK, CELLS_PER_HALFBLOCK); ++i) {
        uint edgeIdx = i * CELLS_PER_HALFBLOCK + localThreadId.x;
        if (edgeIdx >= EDGES_PER_HALFBLOCK) break; // (because these numbers are not perfect multiples)

        uint3 vertexIndices[2] = getGlobalVerticesForEdge(blockIdx3d, edgeIdx, halfBlockIndex);
        float density0 = vertexDensities[to1D(vertexIndices[0], (cb.dimensions + 1) * uint3(1, 1, 1))];
        float density1 = vertexDensities[to1D(vertexIndices[1], (cb.dimensions + 1) * uint3(1, 1, 1))];

        bool needVertex = density0 > ISOVALUE ^ density1 > ISOVALUE;
        // To get an offset into shared memory, each thread can get the number of verts found in waves
        // with lower thread indices using WavePrefixCountBits.
        uint outputVertexIndex = vertexCount + WavePrefixCountBits(needVertex);
        // (Each thread is keeping track of an individual copy of vertexCount, but staying in sync via wave intrinsics) 
        vertexCount += WaveActiveCountBits(needVertex);

        if (!needVertex) continue;
        
        float3 vertexNormals[2] = getVertexNormals(vertexIndices);
        
        float t = interpolateDensity(density0, density1);
        float3 vertPosWorld = cb.minBounds + cb.resolution * lerp(vertexIndices[0], vertexIndices[1], t);
        float4 vertPosClip = mul(float4(vertPosWorld, 1.0), cb.viewProj);
        float3 vertNormal = -normalize(lerp(vertexNormals[0], vertexNormals[1], t)); // Paper negates normals, not sure why!

        // Store the index of the output vertex in shared memory
        // In next step, each thread acts as a cell and will read several vertex indices, so we need them all in shared memory.
        outputVertexIndices[edgeIdx] = outputVertexIndex;
        // Then store all the vertex attributes at this outputVertexIndex
        // (Note, HLSL has a mesh shader restriction: you have to tell it the total output verts + prims there are BEFORE you write to them)
        // (For that reason, mostly, we write the vertex attributes to shared memory first; otherwise we could just write them out right now, before we know the total count).
        vertexWorldPositions[outputVertexIndex] = vertPosWorld;
        vertexClipPositions[outputVertexIndex] = vertPosClip;
        vertexNormalsShared[outputVertexIndex] = vertNormal.xy;
    }

    // Every surface block has surface vertices, but since each workgroup represents a *half*-blocks,
    // it's possible for a workgroup's half block to have no surface vertices. 
    if (vertexCount == 0) return;

    GroupMemoryBarrierWithGroupSync();

    // From here on out, every thread acts as a single cell (TODO: this is where I want to try optimizing; return early for non-surface cells)
    uint localCellIdx1d = globalThreadId.x % CELLS_PER_BLOCK;
    uint3 localCellIdx3d = to3D(localCellIdx1d, CELLS_PER_BLOCK_EDGE * uint3(1, 1, 1)); // TODO: can we avoid this conversion with 3D dispatch?
    uint3 globalCellIdx3d = blockIdx3d * CELLS_PER_BLOCK_EDGE + localCellIdx3d;

    uint mcCase = computeMarchingCubesCase(globalCellIdx3d);
    uint numTris = triangleCounts[mcCase];
    uint triOffset = WavePrefixSum(numTris);
    uint totalTris = WaveActiveSum(numTris);

    // We finally have all the info we need to tell the mesh shader how many vertices and primitives we're outputting.
    SetMeshOutputCounts(vertexCount, totalTris);

    // Now we do the actual marching cubes / outputting of vertices and tris
    for (uint t = 0; t < totalTris; ++t) {
        uint3 triVerts;

        for (uint v = 0; v < 3; ++v) {
            uint cellEdgeIdx = uint(triangleTable[mcCase][t * 3 + v]); // edge index within a cell (0-11)
            uint blockEdgeIdx = cellEdgeToBlockEdge(localCellIdx1d, cellEdgeIdx, halfBlockIndex); // edge within a block (0-169)
            uint outputVertexIndex = outputVertexIndices[blockEdgeIdx];

            triVerts[v] = outputVertexIndex;

            // Write the vertex attributes to the output buffer
            verts[outputVertexIndex].clipPos = vertexClipPositions[outputVertexIndex];
            verts[outputVertexIndex].normal = vertexNormalsShared[outputVertexIndex];
            verts[outputVertexIndex].worldPos = vertexWorldPositions[outputVertexIndex];
        }
        
        // Write the triangle to the output buffer
        triangles[triOffset + t] = triVerts;
    }

}
 