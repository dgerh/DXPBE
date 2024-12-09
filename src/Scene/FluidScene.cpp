#include "FluidScene.h"
#include "SceneConstants.h"

FluidScene::FluidScene(DXContext* context, 
                       RenderPipeline* pipeline, 
                       ComputePipeline* bilevelUniformGridCP, 
                       ComputePipeline* surfaceBlockDetectionCP,
                       ComputePipeline* surfaceCellDetectionCP,
                       ComputePipeline* surfaceVertexCompactionCP,
                       ComputePipeline* surfaceVertexDensityCP,
                       ComputePipeline* surfaceVertexNormalCP,
                       ComputePipeline* bufferClearCP,
                       MeshPipeline* fluidMeshPipeline)
    : Drawable(context, pipeline), 
      bilevelUniformGridCP(bilevelUniformGridCP), 
      surfaceBlockDetectionCP(surfaceBlockDetectionCP),
      surfaceCellDetectionCP(surfaceCellDetectionCP),
      surfaceVertexCompactionCP(surfaceVertexCompactionCP),
      surfaceVertexDensityCP(surfaceVertexDensityCP),
      surfaceVertexNormalCP(surfaceVertexNormalCP),
      bufferClearCP(bufferClearCP),
      fluidMeshPipeline(fluidMeshPipeline)
{
    constructScene();
}

// In this pipeline, drawing is done via a mesh shader
void FluidScene::draw(Camera* camera, unsigned int renderMeshlets) {
    gridConstants.kernelScale = kernelScale;
    gridConstants.kernelRadius = kernelRadius * gridConstants.resolution;
    
    auto cmdList = fluidMeshPipeline->getCommandList();
    MeshShadingConstants meshShadingConstants = { camera->getViewProjMat(), gridConstants.gridDim, gridConstants.resolution, gridConstants.minBounds, renderMeshlets, camera->getPosition(), isovalue };
    cmdList->SetPipelineState(fluidMeshPipeline->getPSO());
    cmdList->SetGraphicsRootSignature(fluidMeshPipeline->getRootSignature());

    // Set descriptor heap
    ID3D12DescriptorHeap* descriptorHeaps[] = { bilevelUniformGridCP->getDescriptorHeap()->Get() };
    cmdList->SetDescriptorHeaps(_countof(descriptorHeaps), descriptorHeaps);

    // Transition surfaceHalfBlockDispatch to an SRV
    D3D12_RESOURCE_BARRIER surfaceHalfBlockDispatchBarrier = CD3DX12_RESOURCE_BARRIER::Transition(
        surfaceHalfBlockDispatch.getBuffer(),
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE|D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE
    );

    // Transition surfaceVertexNormalBuffer to an SRV
    D3D12_RESOURCE_BARRIER surfaceVertexNormalBufferBarrier = CD3DX12_RESOURCE_BARRIER::Transition(
        surfaceVertexNormalBuffer.getBuffer(),
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE|D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE
    );

    // Transition surfaceVertDensityDispatch to a UAV (this is purely for resetting the buffer)
    D3D12_RESOURCE_BARRIER surfaceVertDensityDispatchBarrier = CD3DX12_RESOURCE_BARRIER::Transition(
        surfaceVertDensityDispatch.getBuffer(),
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS
    );
    
    D3D12_RESOURCE_BARRIER barriers[3] = { surfaceVertDensityDispatchBarrier, surfaceHalfBlockDispatchBarrier, surfaceVertexNormalBufferBarrier };
    cmdList->ResourceBarrier(3, barriers);

    // Set graphics root descriptor table
    cmdList->SetGraphicsRootDescriptorTable(0, surfaceBlockIndicesBuffer.getSRVGPUDescriptorHandle());
    cmdList->SetGraphicsRootDescriptorTable(1, surfaceVertDensityBuffer.getSRVGPUDescriptorHandle());
    cmdList->SetGraphicsRootDescriptorTable(2, surfaceVertexNormalBuffer.getSRVGPUDescriptorHandle());
    cmdList->SetGraphicsRootShaderResourceView(3, surfaceHalfBlockDispatch.getGPUVirtualAddress());
    cmdList->SetGraphicsRootUnorderedAccessView(4, surfaceVertDensityDispatch.getGPUVirtualAddress());
    cmdList->SetGraphicsRoot32BitConstants(5, 28, &meshShadingConstants, 0);

    // Transition surfaceHalfBlockDispatch to indirect argument buffer
    D3D12_RESOURCE_BARRIER surfaceHalfBlockDispatchBarrier2 = CD3DX12_RESOURCE_BARRIER::Transition(
        surfaceHalfBlockDispatch.getBuffer(),
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE|D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT
    );

    cmdList->ResourceBarrier(1, &surfaceHalfBlockDispatchBarrier2);

    // Draws
    cmdList->ExecuteIndirect(meshCommandSignature, 1, surfaceHalfBlockDispatch.getBuffer(), 0, nullptr, 0);

    // TODO Temporary: just so these two buffers can be transitioned along with everything else
    D3D12_RESOURCE_BARRIER surfaceHalfBlockDispatchBarrier3 = CD3DX12_RESOURCE_BARRIER::Transition(
        surfaceHalfBlockDispatch.getBuffer(),
        D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE|D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE
    );

    D3D12_RESOURCE_BARRIER surfaceVertDensityDispatchBarrier2 = CD3DX12_RESOURCE_BARRIER::Transition(
        surfaceVertDensityDispatch.getBuffer(),
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE|D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE
    );

    D3D12_RESOURCE_BARRIER barriers2[2] = { surfaceHalfBlockDispatchBarrier3, surfaceVertDensityDispatchBarrier2 };
    cmdList->ResourceBarrier(2, barriers2);
    // End temporary

    transitionBuffers(cmdList, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE|D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    context->executeCommandList(fluidMeshPipeline->getCommandListID());
    context->signalAndWaitForFence(fence, fenceValue);

    context->resetCommandList(fluidMeshPipeline->getCommandListID());
    
    resetBuffers();
}

float getRandomFloatInRange(float min, float max) {
    return min + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (max - min)));
}

void FluidScene::constructScene() {
    int blocksPerEdge = 32;
    float cellWidth = (float)std::max(std::max(GRID_WIDTH, GRID_HEIGHT), GRID_DEPTH) / ((float)blocksPerEdge * (float)CELLS_PER_BLOCK_EDGE);
    gridConstants = { 0, 
                     {blocksPerEdge * CELLS_PER_BLOCK_EDGE, blocksPerEdge * CELLS_PER_BLOCK_EDGE, blocksPerEdge * CELLS_PER_BLOCK_EDGE}, 
                     {0.f, 0.f, 0.f}, 
                     cellWidth, 
                     kernelScale,
                     kernelRadius * cellWidth,
                    };

    // Create cells and blocks buffers
    int numCells = gridConstants.gridDim.x * gridConstants.gridDim.y * gridConstants.gridDim.z;
    int numBlocks = numCells / (CELLS_PER_BLOCK_EDGE * CELLS_PER_BLOCK_EDGE * CELLS_PER_BLOCK_EDGE);
    int numVerts = (gridConstants.gridDim.x + 1) * (gridConstants.gridDim.y + 1) * (gridConstants.gridDim.z + 1);
    
    std::vector<int> cellParticleCounts(numCells, 0);
    std::vector<int> cellParticleIndices(numCells * MAX_PARTICLES_PER_CELL, -1);
    std::vector<Block> blocks(numBlocks);
    std::vector<int> surfaceBlockIndices(numBlocks, 0);
    XMUINT3 dipatchCPU = { 0, 1, 1 };
    std::vector<int> surfaceVertices(numVerts, 0);
    std::vector<int> surfaceVertexIndices(numVerts, 0);
    std::vector<float> surfaceVertexDensities(numVerts, 0.f);
    std::vector<XMFLOAT3> surfaceVertexNormals(numVerts, { 0.f, 0.f, 0.f });

    // Use the descriptor heap for the bilevelUniformGridCP for pretty much everything. Simplifies sharing resources
    blocksBuffer = StructuredBuffer(blocks.data(), numBlocks, sizeof(Block));
    blocksBuffer.passDataToGPU(*context, bilevelUniformGridCP->getCommandList(), bilevelUniformGridCP->getCommandListID());
    blocksBuffer.createUAV(*context, bilevelUniformGridCP->getDescriptorHeap());
    blocksBuffer.createSRV(*context, bilevelUniformGridCP->getDescriptorHeap());

    cellParticleCountBuffer = StructuredBuffer(cellParticleCounts.data(), numCells, sizeof(int));
    cellParticleCountBuffer.passDataToGPU(*context, bilevelUniformGridCP->getCommandList(), bilevelUniformGridCP->getCommandListID());
    cellParticleCountBuffer.createUAV(*context, bilevelUniformGridCP->getDescriptorHeap());
    cellParticleCountBuffer.createSRV(*context, bilevelUniformGridCP->getDescriptorHeap());

    cellParticleIndicesBuffer = StructuredBuffer(cellParticleIndices.data(), MAX_PARTICLES_PER_CELL * numCells, sizeof(int));
    cellParticleIndicesBuffer.passDataToGPU(*context, bilevelUniformGridCP->getCommandList(), bilevelUniformGridCP->getCommandListID());
    cellParticleIndicesBuffer.createUAV(*context, bilevelUniformGridCP->getDescriptorHeap());
    cellParticleIndicesBuffer.createSRV(*context, bilevelUniformGridCP->getDescriptorHeap());

    surfaceBlockIndicesBuffer = StructuredBuffer(surfaceBlockIndices.data(), numBlocks, sizeof(unsigned int));
    surfaceBlockIndicesBuffer.passDataToGPU(*context, bilevelUniformGridCP->getCommandList(), bilevelUniformGridCP->getCommandListID());
    surfaceBlockIndicesBuffer.createUAV(*context, bilevelUniformGridCP->getDescriptorHeap());
    surfaceBlockIndicesBuffer.createSRV(*context, bilevelUniformGridCP->getDescriptorHeap());

    surfaceBlockDispatch = StructuredBuffer(&dipatchCPU, 1, sizeof(XMUINT3));
    surfaceBlockDispatch.passDataToGPU(*context, bilevelUniformGridCP->getCommandList(), bilevelUniformGridCP->getCommandListID());
    surfaceBlockDispatch.createUAV(*context, bilevelUniformGridCP->getDescriptorHeap());
    surfaceBlockDispatch.createSRV(*context, bilevelUniformGridCP->getDescriptorHeap());

    surfaceHalfBlockDispatch = StructuredBuffer(&dipatchCPU, 1, sizeof(XMUINT3));
    surfaceHalfBlockDispatch.passDataToGPU(*context, bilevelUniformGridCP->getCommandList(), bilevelUniformGridCP->getCommandListID());
    surfaceHalfBlockDispatch.createUAV(*context, bilevelUniformGridCP->getDescriptorHeap());
    surfaceHalfBlockDispatch.createSRV(*context, bilevelUniformGridCP->getDescriptorHeap());

    surfaceVerticesBuffer = StructuredBuffer(surfaceVertices.data(), numVerts, sizeof(unsigned int));
    surfaceVerticesBuffer.passDataToGPU(*context, bilevelUniformGridCP->getCommandList(), bilevelUniformGridCP->getCommandListID());
    surfaceVerticesBuffer.createUAV(*context, bilevelUniformGridCP->getDescriptorHeap());
    surfaceVerticesBuffer.createSRV(*context, bilevelUniformGridCP->getDescriptorHeap());

    surfaceVertexIndicesBuffer = StructuredBuffer(surfaceVertexIndices.data(), numVerts, sizeof(unsigned int));
    surfaceVertexIndicesBuffer.passDataToGPU(*context, bilevelUniformGridCP->getCommandList(), bilevelUniformGridCP->getCommandListID());
    surfaceVertexIndicesBuffer.createUAV(*context, bilevelUniformGridCP->getDescriptorHeap());
    surfaceVertexIndicesBuffer.createSRV(*context, bilevelUniformGridCP->getDescriptorHeap());

    surfaceVertDensityDispatch = StructuredBuffer(&dipatchCPU, 1, sizeof(XMUINT3));
    surfaceVertDensityDispatch.passDataToGPU(*context, bilevelUniformGridCP->getCommandList(), bilevelUniformGridCP->getCommandListID());
    surfaceVertDensityDispatch.createUAV(*context, bilevelUniformGridCP->getDescriptorHeap());
    surfaceVertDensityDispatch.createSRV(*context, bilevelUniformGridCP->getDescriptorHeap());

    surfaceVertDensityBuffer = StructuredBuffer(surfaceVertexDensities.data(), numVerts, sizeof(float));
    surfaceVertDensityBuffer.passDataToGPU(*context, bilevelUniformGridCP->getCommandList(), bilevelUniformGridCP->getCommandListID());
    surfaceVertDensityBuffer.createUAV(*context, bilevelUniformGridCP->getDescriptorHeap());
    surfaceVertDensityBuffer.createSRV(*context, bilevelUniformGridCP->getDescriptorHeap());

    surfaceVertexNormalBuffer = StructuredBuffer(surfaceVertexNormals.data(), numVerts, sizeof(XMFLOAT3));
    surfaceVertexNormalBuffer.passDataToGPU(*context, bilevelUniformGridCP->getCommandList(), bilevelUniformGridCP->getCommandListID());
    surfaceVertexNormalBuffer.createUAV(*context, bilevelUniformGridCP->getDescriptorHeap()); 
    surfaceVertexNormalBuffer.createSRV(*context, bilevelUniformGridCP->getDescriptorHeap());

	// Create Command Signature
	// Describe the arguments for an indirect dispatch
	D3D12_INDIRECT_ARGUMENT_DESC argumentDesc = {};
	argumentDesc.Type = D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH;

	// Command signature description
	D3D12_COMMAND_SIGNATURE_DESC commandSignatureDesc = {};
	commandSignatureDesc.ByteStride = sizeof(XMUINT3); // Argument buffer stride
	commandSignatureDesc.NumArgumentDescs = 1; // One argument descriptor
	commandSignatureDesc.pArgumentDescs = &argumentDesc;

	// Create the command signature
	context->getDevice()->CreateCommandSignature(&commandSignatureDesc, nullptr, IID_PPV_ARGS(&commandSignature));

    // Command signature for mesh shader
    D3D12_INDIRECT_ARGUMENT_DESC argumentDescMesh = {};
    argumentDescMesh.Type = D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH_MESH;

    D3D12_COMMAND_SIGNATURE_DESC commandSignatureDescMesh = {};
    commandSignatureDescMesh.ByteStride = sizeof(XMUINT3); // Argument buffer stride
    commandSignatureDescMesh.NumArgumentDescs = 1; // One argument descriptor
    commandSignatureDescMesh.pArgumentDescs = &argumentDescMesh;

    context->getDevice()->CreateCommandSignature(&commandSignatureDescMesh, nullptr, IID_PPV_ARGS(&meshCommandSignature));

    // Create fence
    context->getDevice()->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));

    // Transition all resources to UAVs to start
    transitionBuffers(bilevelUniformGridCP->getCommandList(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    context->executeCommandList(bilevelUniformGridCP->getCommandListID());
    context->signalAndWaitForFence(fence, fenceValue);
    context->resetCommandList(bilevelUniformGridCP->getCommandListID());
}

void FluidScene::compute(
    StructuredBuffer* pbmpmPositionsBuffer,
    int numParticles
) {
    gridConstants.numParticles = numParticles;
    positionBuffer = pbmpmPositionsBuffer;

    computeBilevelUniformGrid();
    computeSurfaceBlockDetection();
    computeSurfaceCellDetection();
    compactSurfaceVertices();
    computeSurfaceVertexDensity();
    computeSurfaceVertexNormal();
}

void FluidScene::computeBilevelUniformGrid() {
    auto cmdList = bilevelUniformGridCP->getCommandList();

    cmdList->SetPipelineState(bilevelUniformGridCP->getPSO());
    cmdList->SetComputeRootSignature(bilevelUniformGridCP->getRootSignature());

    // Set descriptor heap
    ID3D12DescriptorHeap* computeDescriptorHeaps[] = { bilevelUniformGridCP->getDescriptorHeap()->Get() };
    cmdList->SetDescriptorHeaps(_countof(computeDescriptorHeaps), computeDescriptorHeaps);

    // Set compute root descriptor table
    cmdList->SetComputeRootDescriptorTable(0, positionBuffer->getSRVGPUDescriptorHandle());
    cmdList->SetComputeRootDescriptorTable(1, cellParticleCountBuffer.getUAVGPUDescriptorHandle());
    cmdList->SetComputeRootDescriptorTable(2, cellParticleIndicesBuffer.getUAVGPUDescriptorHandle());
    cmdList->SetComputeRootDescriptorTable(3, blocksBuffer.getUAVGPUDescriptorHandle());
    cmdList->SetComputeRoot32BitConstants(4, 10, &gridConstants, 0);

    // Dispatch
    int numWorkGroups = (gridConstants.numParticles + BILEVEL_UNIFORM_GRID_THREADS_X - 1) / BILEVEL_UNIFORM_GRID_THREADS_X;
    cmdList->Dispatch(numWorkGroups, 1, 1);

    // Transition blocksBuffer from UAV to SRV for the next pass
    D3D12_RESOURCE_BARRIER blocksBufferBarrier = CD3DX12_RESOURCE_BARRIER::Transition(
        blocksBuffer.getBuffer(),
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
    );

    cmdList->ResourceBarrier(1, &blocksBufferBarrier);

    // Execute command list
    context->executeCommandList(bilevelUniformGridCP->getCommandListID());
    context->signalAndWaitForFence(fence, fenceValue);

    // Reinitialize command list
    context->resetCommandList(bilevelUniformGridCP->getCommandListID());
}

void FluidScene::computeSurfaceBlockDetection() {
    auto cmdList = surfaceBlockDetectionCP->getCommandList();

    cmdList->SetPipelineState(surfaceBlockDetectionCP->getPSO());
    cmdList->SetComputeRootSignature(surfaceBlockDetectionCP->getRootSignature());

    // Set descriptor heap
    ID3D12DescriptorHeap* computeDescriptorHeaps[] = { bilevelUniformGridCP->getDescriptorHeap()->Get() };
    cmdList->SetDescriptorHeaps(_countof(computeDescriptorHeaps), computeDescriptorHeaps);

    int numCells = gridConstants.gridDim.x * gridConstants.gridDim.y * gridConstants.gridDim.z;
    int numBlocks = numCells / (CELLS_PER_BLOCK_EDGE * CELLS_PER_BLOCK_EDGE * CELLS_PER_BLOCK_EDGE);

    // Set compute root descriptor table
    cmdList->SetComputeRootDescriptorTable(0, blocksBuffer.getSRVGPUDescriptorHandle());
    cmdList->SetComputeRootDescriptorTable(1, surfaceBlockIndicesBuffer.getUAVGPUDescriptorHandle());
    cmdList->SetComputeRootUnorderedAccessView(2, surfaceBlockDispatch.getGPUVirtualAddress());
    cmdList->SetComputeRoot32BitConstants(3, 1, &numBlocks, 0);

    // Dispatch
    int numWorkGroups = (numBlocks + SURFACE_BLOCK_DETECTION_THREADS_X - 1) / SURFACE_BLOCK_DETECTION_THREADS_X;
    cmdList->Dispatch(numWorkGroups, 1, 1);

    context->executeCommandList(surfaceBlockDetectionCP->getCommandListID());
    context->signalAndWaitForFence(fence, fenceValue);

    context->resetCommandList(surfaceBlockDetectionCP->getCommandListID());
}

void FluidScene::computeSurfaceCellDetection() {
    auto cmdList = surfaceCellDetectionCP->getCommandList();

    cmdList->SetPipelineState(surfaceCellDetectionCP->getPSO());
    cmdList->SetComputeRootSignature(surfaceCellDetectionCP->getRootSignature());

    // Set descriptor heap
    ID3D12DescriptorHeap* computeDescriptorHeaps[] = { bilevelUniformGridCP->getDescriptorHeap()->Get() };
    cmdList->SetDescriptorHeaps(_countof(computeDescriptorHeaps), computeDescriptorHeaps);

    // Transition surfaceBlockIndicesBuffer to SRV 
    D3D12_RESOURCE_BARRIER surfaceBlockIndicesBufferBarrier = CD3DX12_RESOURCE_BARRIER::Transition(
        surfaceBlockIndicesBuffer.getBuffer(),
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
    );

    // Transition surfaceBlockDispatch to an SRV
    D3D12_RESOURCE_BARRIER surfaceBlockDispatchBarrier = CD3DX12_RESOURCE_BARRIER::Transition(
        surfaceBlockDispatch.getBuffer(),
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
    );

    // Transition cellsBuffer to SRV 
    D3D12_RESOURCE_BARRIER cellParticleCountBufferBarrier = CD3DX12_RESOURCE_BARRIER::Transition(
        cellParticleCountBuffer.getBuffer(),
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
    );

    D3D12_RESOURCE_BARRIER barriers[3] = { surfaceBlockIndicesBufferBarrier, surfaceBlockDispatchBarrier, cellParticleCountBufferBarrier };
    cmdList->ResourceBarrier(3, barriers);

    // Set compute root descriptor table
    cmdList->SetComputeRootDescriptorTable(0, surfaceBlockIndicesBuffer.getSRVGPUDescriptorHandle());
    cmdList->SetComputeRootDescriptorTable(1, cellParticleCountBuffer.getSRVGPUDescriptorHandle());
    cmdList->SetComputeRootDescriptorTable(2, surfaceVerticesBuffer.getUAVGPUDescriptorHandle());
    cmdList->SetComputeRootShaderResourceView(3, surfaceBlockDispatch.getGPUVirtualAddress());
    cmdList->SetComputeRootUnorderedAccessView(4, surfaceHalfBlockDispatch.getGPUVirtualAddress());
    cmdList->SetComputeRoot32BitConstants(5, 10, &gridConstants, 0);

    // Transition surfaceBlockDispatch to indirect argument buffer
    D3D12_RESOURCE_BARRIER surfaceBlockDispatchBarrier2 = CD3DX12_RESOURCE_BARRIER::Transition(
        surfaceBlockDispatch.getBuffer(),
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT
    );

    cmdList->ResourceBarrier(1, &surfaceBlockDispatchBarrier2);

    // Dispatch
    cmdList->ExecuteIndirect(commandSignature, 1, surfaceBlockDispatch.getBuffer(), 0, nullptr, 0);

    context->executeCommandList(surfaceCellDetectionCP->getCommandListID());
    context->signalAndWaitForFence(fence, fenceValue);

    context->resetCommandList(surfaceCellDetectionCP->getCommandListID());
}

void FluidScene::compactSurfaceVertices() {
    auto cmdList = surfaceVertexCompactionCP->getCommandList();

    cmdList->SetPipelineState(surfaceVertexCompactionCP->getPSO());
    cmdList->SetComputeRootSignature(surfaceVertexCompactionCP->getRootSignature());

    // Set descriptor heap
    ID3D12DescriptorHeap* computeDescriptorHeaps[] = { bilevelUniformGridCP->getDescriptorHeap()->Get() };
    cmdList->SetDescriptorHeaps(_countof(computeDescriptorHeaps), computeDescriptorHeaps);

    // Transition surfaceVerticesBuffer back to SRV 
    D3D12_RESOURCE_BARRIER surfaceVerticesBufferBarrier = CD3DX12_RESOURCE_BARRIER::Transition(
        surfaceVerticesBuffer.getBuffer(),
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE 
    );

    cmdList->ResourceBarrier(1, &surfaceVerticesBufferBarrier);

    // Set compute root descriptor table
    cmdList->SetComputeRootDescriptorTable(0, surfaceVerticesBuffer.getSRVGPUDescriptorHandle());
    cmdList->SetComputeRootDescriptorTable(1, surfaceVertexIndicesBuffer.getUAVGPUDescriptorHandle());
    cmdList->SetComputeRootUnorderedAccessView(2, surfaceVertDensityDispatch.getGPUVirtualAddress());
    cmdList->SetComputeRoot32BitConstants(3, 10, &gridConstants, 0);

    // Dispatch
    int numVertices = (gridConstants.gridDim.x + 1) * (gridConstants.gridDim.y + 1) * (gridConstants.gridDim.z + 1);
    int numWorkGroups = (numVertices + SURFACE_VERTEX_COMPACTION_THREADS_X - 1) / SURFACE_VERTEX_COMPACTION_THREADS_X;
        cmdList->Dispatch(numWorkGroups, 1, 1);

    context->executeCommandList(surfaceVertexCompactionCP->getCommandListID());
    context->signalAndWaitForFence(fence, fenceValue);

    context->resetCommandList(surfaceVertexCompactionCP->getCommandListID());
}

void FluidScene::computeSurfaceVertexDensity() {
    auto cmdList = surfaceVertexDensityCP->getCommandList();

    cmdList->SetPipelineState(surfaceVertexDensityCP->getPSO());
    cmdList->SetComputeRootSignature(surfaceVertexDensityCP->getRootSignature());

    // Set descriptor heap
    ID3D12DescriptorHeap* computeDescriptorHeaps[] = { bilevelUniformGridCP->getDescriptorHeap()->Get() };
    cmdList->SetDescriptorHeaps(_countof(computeDescriptorHeaps), computeDescriptorHeaps);

    // Transition surfaceVertexIndicesBuffer to SRV
    D3D12_RESOURCE_BARRIER surfaceVertexIndicesBufferBarrier = CD3DX12_RESOURCE_BARRIER::Transition(
        surfaceVertexIndicesBuffer.getBuffer(),
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
    );

    // Transition surfaceVertDensityDispatch to an SRV
    D3D12_RESOURCE_BARRIER surfaceVertDensityDispatchBarrier = CD3DX12_RESOURCE_BARRIER::Transition(
        surfaceVertDensityDispatch.getBuffer(),
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
    );

    // Transition surfaceBlockDispatch to a UAV
    D3D12_RESOURCE_BARRIER surfaceBlockDispatchBarrier = CD3DX12_RESOURCE_BARRIER::Transition(
        surfaceBlockDispatch.getBuffer(),
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS
    );

    D3D12_RESOURCE_BARRIER barriers[3] = { surfaceVertexIndicesBufferBarrier, surfaceVertDensityDispatchBarrier, surfaceBlockDispatchBarrier };
    cmdList->ResourceBarrier(3, barriers);

    // Set compute root descriptor table
    cmdList->SetComputeRootDescriptorTable(0, positionBuffer->getSRVGPUDescriptorHandle());
    cmdList->SetComputeRootDescriptorTable(1, cellParticleCountBuffer.getSRVGPUDescriptorHandle());
    cmdList->SetComputeRootDescriptorTable(2, cellParticleIndicesBuffer.getSRVGPUDescriptorHandle());
    cmdList->SetComputeRootDescriptorTable(3, surfaceVertexIndicesBuffer.getSRVGPUDescriptorHandle());
    cmdList->SetComputeRootShaderResourceView(4, surfaceVertDensityDispatch.getGPUVirtualAddress());
    cmdList->SetComputeRootUnorderedAccessView(5, surfaceBlockDispatch.getGPUVirtualAddress());
    cmdList->SetComputeRootDescriptorTable(6, surfaceVertDensityBuffer.getUAVGPUDescriptorHandle());
    cmdList->SetComputeRoot32BitConstants(7, 10, &gridConstants, 0);

    // Transition surfaceVertDensityDispatch to indirect argument buffer
    D3D12_RESOURCE_BARRIER surfaceVertDensityDispatchBarrier2 = CD3DX12_RESOURCE_BARRIER::Transition(
        surfaceVertDensityDispatch.getBuffer(),
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT
    );

    cmdList->ResourceBarrier(1, &surfaceVertDensityDispatchBarrier2);

    // Dispatch
    cmdList->ExecuteIndirect(commandSignature, 1, surfaceVertDensityDispatch.getBuffer(), 0, nullptr, 0);

    context->executeCommandList(surfaceVertexDensityCP->getCommandListID());
    context->signalAndWaitForFence(fence, fenceValue);

    context->resetCommandList(surfaceVertexDensityCP->getCommandListID());
}

void FluidScene::computeSurfaceVertexNormal() {
    auto cmdList = surfaceVertexNormalCP->getCommandList();

    cmdList->SetPipelineState(surfaceVertexNormalCP->getPSO());
    cmdList->SetComputeRootSignature(surfaceVertexNormalCP->getRootSignature());

    // Set descriptor heap
    ID3D12DescriptorHeap* computeDescriptorHeaps[] = { bilevelUniformGridCP->getDescriptorHeap()->Get() };
    cmdList->SetDescriptorHeaps(_countof(computeDescriptorHeaps), computeDescriptorHeaps);

    // Transition surfaceVertDensityBuffer to SRV
    D3D12_RESOURCE_BARRIER surfaceVertDensityBufferBarrier = CD3DX12_RESOURCE_BARRIER::Transition(
        surfaceVertDensityBuffer.getBuffer(),
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
    );

    // Transition surfaceVertDensityDispatch to an SRV
    D3D12_RESOURCE_BARRIER surfaceVertDensityDispatchBarrier = CD3DX12_RESOURCE_BARRIER::Transition(
        surfaceVertDensityDispatch.getBuffer(),
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
    );

    D3D12_RESOURCE_BARRIER barriers[2] = { surfaceVertDensityBufferBarrier, surfaceVertDensityDispatchBarrier };
    cmdList->ResourceBarrier(2, barriers);

    // Set compute root descriptor table
    cmdList->SetComputeRootDescriptorTable(0, surfaceVertDensityBuffer.getSRVGPUDescriptorHandle());
    cmdList->SetComputeRootDescriptorTable(1, surfaceVertexIndicesBuffer.getSRVGPUDescriptorHandle());
    cmdList->SetComputeRootShaderResourceView(2, surfaceVertDensityDispatch.getGPUVirtualAddress());
    cmdList->SetComputeRootDescriptorTable(3, surfaceVertexNormalBuffer.getUAVGPUDescriptorHandle());
    cmdList->SetComputeRoot32BitConstants(4, 10, &gridConstants, 0);

    // Transition surfaceVertDensityDispatch to indirect argument buffer
    D3D12_RESOURCE_BARRIER surfaceVertDensityDispatchBarrier2 = CD3DX12_RESOURCE_BARRIER::Transition(
        surfaceVertDensityDispatch.getBuffer(),
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT
    );

    cmdList->ResourceBarrier(1, &surfaceVertDensityDispatchBarrier2);

    // Dispatch
    cmdList->ExecuteIndirect(commandSignature, 1, surfaceVertDensityDispatch.getBuffer(), 0, nullptr, 0);

    context->executeCommandList(surfaceVertexNormalCP->getCommandListID());
    context->signalAndWaitForFence(fence, fenceValue);

    context->resetCommandList(surfaceVertexNormalCP->getCommandListID());
}

void FluidScene::releaseResources() {
    renderPipeline->releaseResources();
    bilevelUniformGridCP->releaseResources();
    surfaceBlockDetectionCP->releaseResources();
    surfaceCellDetectionCP->releaseResources();
    cellParticleCountBuffer.releaseResources();
    cellParticleIndicesBuffer.releaseResources();
    blocksBuffer.releaseResources();
    surfaceBlockIndicesBuffer.releaseResources();
    surfaceBlockDispatch.releaseResources();
    surfaceHalfBlockDispatch.releaseResources();
    surfaceVerticesBuffer.releaseResources();
    surfaceVertexIndicesBuffer.releaseResources();
    surfaceVertDensityDispatch.releaseResources();
    surfaceVertDensityBuffer.releaseResources();
    surfaceVertexNormalBuffer.releaseResources();
}

void FluidScene::transitionBuffers(ID3D12GraphicsCommandList6* cmdList, D3D12_RESOURCE_STATES beforeState, D3D12_RESOURCE_STATES afterState) {
    D3D12_RESOURCE_BARRIER cellParticleCountsBufferBarrier = CD3DX12_RESOURCE_BARRIER::Transition(
        cellParticleCountBuffer.getBuffer(),
        beforeState,
        afterState
    );

    D3D12_RESOURCE_BARRIER cellParticleIndicesBufferBarrier = CD3DX12_RESOURCE_BARRIER::Transition(
        cellParticleIndicesBuffer.getBuffer(),
        beforeState,
        afterState
    );

    D3D12_RESOURCE_BARRIER blocksBufferBarrier = CD3DX12_RESOURCE_BARRIER::Transition(
        blocksBuffer.getBuffer(),
        beforeState,
        afterState
    );
    
    D3D12_RESOURCE_BARRIER surfaceBlockIndicesBufferBarrier = CD3DX12_RESOURCE_BARRIER::Transition(
        surfaceBlockIndicesBuffer.getBuffer(),
        beforeState,
        afterState
    );

    D3D12_RESOURCE_BARRIER surfaceVerticesBufferBarrier = CD3DX12_RESOURCE_BARRIER::Transition(
        surfaceVerticesBuffer.getBuffer(),
        beforeState,
        afterState
    );

    D3D12_RESOURCE_BARRIER surfaceVertexIndicesBufferBarrier = CD3DX12_RESOURCE_BARRIER::Transition(
        surfaceVertexIndicesBuffer.getBuffer(),
        beforeState,
        afterState
    );

    D3D12_RESOURCE_BARRIER surfaceVertDensityBufferBarrier = CD3DX12_RESOURCE_BARRIER::Transition(
        surfaceVertDensityBuffer.getBuffer(),
        beforeState,
        afterState
    );

    D3D12_RESOURCE_BARRIER surfaceVertexNormalBufferBarrier2 = CD3DX12_RESOURCE_BARRIER::Transition(
        surfaceVertexNormalBuffer.getBuffer(),
        beforeState,
        afterState
    );

    D3D12_RESOURCE_BARRIER surfaceBlockDispatchBarrier = CD3DX12_RESOURCE_BARRIER::Transition(
        surfaceBlockDispatch.getBuffer(),
        beforeState,
        afterState
    );

    D3D12_RESOURCE_BARRIER surfaceHalfBlockDispatchBarrier = CD3DX12_RESOURCE_BARRIER::Transition(
        surfaceHalfBlockDispatch.getBuffer(),
        beforeState,
        afterState
    );

    D3D12_RESOURCE_BARRIER surfaceVertDensityDispatchBarrier = CD3DX12_RESOURCE_BARRIER::Transition(
        surfaceVertDensityDispatch.getBuffer(),
        beforeState,
        afterState
    );

    D3D12_RESOURCE_BARRIER barriers[11] = { 
        cellParticleCountsBufferBarrier,
        cellParticleIndicesBufferBarrier,
        blocksBufferBarrier,
        surfaceBlockIndicesBufferBarrier,
        surfaceVerticesBufferBarrier,
        surfaceVertexIndicesBufferBarrier, 
        surfaceVertDensityBufferBarrier, 
        surfaceVertexNormalBufferBarrier2, 
        surfaceBlockDispatchBarrier, 
        surfaceHalfBlockDispatchBarrier, 
        surfaceVertDensityDispatchBarrier 
    };

    cmdList->ResourceBarrier(11, barriers);
}

void FluidScene::resetBuffers() {
	constexpr UINT THREAD_GROUP_SIZE = 256;
    int numCells = gridConstants.gridDim.x * gridConstants.gridDim.y * gridConstants.gridDim.z;
    int numCellIndices = numCells * MAX_PARTICLES_PER_CELL;
    int numBlocks = numCells / (CELLS_PER_BLOCK_EDGE * CELLS_PER_BLOCK_EDGE * CELLS_PER_BLOCK_EDGE);
    int numVerts = (gridConstants.gridDim.x + 1) * (gridConstants.gridDim.y + 1) * (gridConstants.gridDim.z + 1);

	// Bind the PSO and Root Signature
    auto cmdList = bufferClearCP->getCommandList();
    cmdList->SetPipelineState(bufferClearCP->getPSO());
    cmdList->SetComputeRootSignature(bufferClearCP->getRootSignature());

    // Bind the descriptor heap
	ID3D12DescriptorHeap* descriptorHeaps[] = { bilevelUniformGridCP->getDescriptorHeap()->Get() };
	cmdList->SetDescriptorHeaps(_countof(descriptorHeaps), descriptorHeaps);

    cmdList->SetComputeRoot32BitConstants(0, 1, &numCells, 0);
	cmdList->SetComputeRootDescriptorTable(1, cellParticleCountBuffer.getUAVGPUDescriptorHandle());
	cmdList->Dispatch((numCells + THREAD_GROUP_SIZE - 1) / THREAD_GROUP_SIZE, 1, 1);

    cmdList->SetComputeRoot32BitConstants(0, 1, &numCellIndices, 0);
    cmdList->SetComputeRootDescriptorTable(1, cellParticleIndicesBuffer.getUAVGPUDescriptorHandle());
    cmdList->Dispatch((numCellIndices + THREAD_GROUP_SIZE - 1) / THREAD_GROUP_SIZE, 1, 1);

    cmdList->SetComputeRoot32BitConstants(0, 1, &numBlocks, 0);
    cmdList->SetComputeRootDescriptorTable(1, blocksBuffer.getUAVGPUDescriptorHandle());
    cmdList->Dispatch((numBlocks + THREAD_GROUP_SIZE - 1) / THREAD_GROUP_SIZE, 1, 1);

    cmdList->SetComputeRoot32BitConstants(0, 1, &numVerts, 0);
    cmdList->SetComputeRootDescriptorTable(1, surfaceVerticesBuffer.getUAVGPUDescriptorHandle());
    cmdList->Dispatch((numVerts + THREAD_GROUP_SIZE - 1) / THREAD_GROUP_SIZE, 1, 1);

    cmdList->SetComputeRoot32BitConstants(0, 1, &numVerts, 0);
    cmdList->SetComputeRootDescriptorTable(1, surfaceVertexIndicesBuffer.getUAVGPUDescriptorHandle());
    cmdList->Dispatch((numVerts + THREAD_GROUP_SIZE - 1) / THREAD_GROUP_SIZE, 1, 1);

    cmdList->SetComputeRoot32BitConstants(0, 1, &numVerts, 0);
    cmdList->SetComputeRootDescriptorTable(1, surfaceVertDensityBuffer.getUAVGPUDescriptorHandle());
    cmdList->Dispatch((numVerts + THREAD_GROUP_SIZE - 1) / THREAD_GROUP_SIZE, 1, 1);

    context->executeCommandList(bufferClearCP->getCommandListID());
    context->signalAndWaitForFence(fence, fenceValue);
    context->resetCommandList(bufferClearCP->getCommandListID());
}