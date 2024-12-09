#pragma once

static const float PI = 3.1415926f;

/* ================== Constants for the mesh shading pipeline ================== */ 

static const int MAX_PARTICLES_PER_CELL = 16;
// This really should not be changed. Mesh shaders have very hard limits on how many verts and primitives they can output.
// If this is made any bigger, the mesh shader will have vertex overflow.
static const int CELLS_PER_BLOCK_EDGE = 4; // each block has 4x4x4 cells
static const int CELLS_PER_BLOCK = CELLS_PER_BLOCK_EDGE * CELLS_PER_BLOCK_EDGE * CELLS_PER_BLOCK_EDGE;
static const int CELLS_PER_HALFBLOCK = CELLS_PER_BLOCK / 2;
static const int HALFBLOCK_CELLS_X = CELLS_PER_BLOCK_EDGE;
static const int HALFBLOCK_CELLS_Y = CELLS_PER_BLOCK_EDGE;
static const int HALFBLOCK_CELLS_Z = CELLS_PER_BLOCK_EDGE / 2;
static const int FILLED_BLOCK = (CELLS_PER_BLOCK_EDGE + 2) * (CELLS_PER_BLOCK_EDGE + 2) * (CELLS_PER_BLOCK_EDGE + 2);
static const int BILEVEL_UNIFORM_GRID_THREADS_X = 64;
static const int SURFACE_BLOCK_DETECTION_THREADS_X = 64;
// For this compute pass, its important for the workgroup size to match the number of cells per block,
// because we use shared memory in this pass and its most coherent if workgroups map to blocks.
static const int SURFACE_CELL_DETECTION_THREADS = CELLS_PER_BLOCK_EDGE;
static const int SURFACE_VERTEX_COMPACTION_THREADS_X = 64;
// It's actually not so important for the workgroup size for this pass to match the number of vertices per block. By this point
// we no longer have block-level coherency of vertices anyway. But use this as a starting point - can adjust it later.
static const int SURFACE_VERTEX_DENSITY_THREADS_X = 32;
static const int SURFACE_VERTEX_NORMAL_THREADS_X = 16;
// Given the method of mesh shading half-blocks, we can place these upper limits on the outputs of each mesh shader workgroup:
static const float EPSILON = 0.00001f;
static const int MAX_PRIMITIVES = 128;
static const int MAX_VERTICES = 170;
static const int EDGES_PER_HALFBLOCK = 170; // equal to max verts; just giving it a new name for clarity

// This is just for shaders, types will not compile in C++ code
#ifndef __cplusplus
struct BilevelUniformGridConstants {
    int numParticles;
    int3 dimensions;
    float3 minBounds;
    float resolution;
    float kernelScale;
    float kernelRadius;
};

struct MeshShadingConstants {
    float4x4 viewProj;
    int3 dimensions;
    float resolution;
    float3 minBounds;
    unsigned int renderMeshlets;
    float3 cameraPos;
    float isovalue;
};
#endif