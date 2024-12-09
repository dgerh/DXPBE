#pragma once
#include <vector>
#include "Drawable.h"
#include "../D3D/Pipeline/ComputePipeline.h"
#include "../D3D/Pipeline/MeshPipeline.h"
#include "../D3D/StructuredBuffer.h"
#include "../Shaders/constants.h"

struct GridConstants {
    int numParticles;
    XMINT3 gridDim;
    XMFLOAT3 minBounds;
    float resolution;
    float kernelScale;
    float kernelRadius;
};

// TODO: can just combine this with grid constants
struct MeshShadingConstants {
    XMMATRIX viewProj;
    XMINT3 dimensions;
    float resolution;
    XMFLOAT3 minBounds;
    unsigned int renderMeshlets;
    XMFLOAT3 cameraPos;
    float isovalue;
};

struct Cell {
    int particleCount;
    int particleIndices[MAX_PARTICLES_PER_CELL];
};

struct Block {
    int nonEmptyCellCount;
};

class FluidScene : public Drawable {
public:
    FluidScene() = delete;
    FluidScene(DXContext* context, 
               RenderPipeline *pipeline, 
               ComputePipeline* bilevelUniformGridCP, 
               ComputePipeline* surfaceBlockDetectionCP,
               ComputePipeline* surfaceCellDetectionCP,
               ComputePipeline* surfaceVertexCompactionCP,
               ComputePipeline* surfaceVertexDensityCP,
               ComputePipeline* surfaceVertexNormalCP,
               ComputePipeline* bufferClearCP,
               MeshPipeline* fluidMeshPipeline);

    void compute(
        StructuredBuffer* positionsBuffer,
        int numParticles
    );
    void draw(Camera* camera, unsigned int renderMeshlets);
    void constructScene();
    void computeBilevelUniformGrid();
    void computeSurfaceBlockDetection();
    void computeSurfaceCellDetection();
    void compactSurfaceVertices();
    void computeSurfaceVertexDensity();
    void computeSurfaceVertexNormal();
    void releaseResources();

    float* getIsovalue() { return &isovalue; }
    float* getKernelScale() { return &kernelScale; }
    float* getKernelRadius() { return &kernelRadius; }

private:
    void transitionBuffers(ID3D12GraphicsCommandList6* cmdList, D3D12_RESOURCE_STATES beforeState, D3D12_RESOURCE_STATES afterState);
    void resetBuffers();

    GridConstants gridConstants;
    
    ComputePipeline* bilevelUniformGridCP;
    ComputePipeline* surfaceBlockDetectionCP;
    ComputePipeline* surfaceCellDetectionCP;
    ComputePipeline* surfaceVertexCompactionCP;
    ComputePipeline* surfaceVertexDensityCP;
    ComputePipeline* surfaceVertexNormalCP;
    ComputePipeline* bufferClearCP;

    MeshPipeline* fluidMeshPipeline;
    
    UINT64 fenceValue = 1;
	ComPointer<ID3D12Fence> fence;

	ID3D12CommandSignature* commandSignature = nullptr;
    ID3D12CommandSignature* meshCommandSignature = nullptr;

    StructuredBuffer* positionBuffer;
    StructuredBuffer cellParticleCountBuffer;
    StructuredBuffer cellParticleIndicesBuffer;
    StructuredBuffer blocksBuffer;
    StructuredBuffer surfaceBlockIndicesBuffer;
    StructuredBuffer surfaceBlockDispatch;
    StructuredBuffer surfaceHalfBlockDispatch; // This is just 2x surfaceBlockDispatch, but saves us a round trip to the GPU to multiply by 2
    StructuredBuffer surfaceVerticesBuffer;
    StructuredBuffer surfaceVertexIndicesBuffer;
    StructuredBuffer surfaceVertDensityDispatch;
    StructuredBuffer surfaceVertDensityBuffer;
    StructuredBuffer surfaceVertexNormalBuffer;

    float isovalue{ 0.03f };
    float kernelScale{ 4.0f };
    float kernelRadius{ 1.01f };
};