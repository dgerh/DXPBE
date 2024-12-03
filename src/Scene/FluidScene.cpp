#include "FluidScene.h"

FluidScene::FluidScene(DXContext* context, 
                       RenderPipeline* pipeline, 
                       ComputePipeline* bilevelUniformGridCP, 
                       ComputePipeline* surfaceBlockDetectionCP,
                       ComputePipeline* surfaceCellDetectionCP,
                       ComputePipeline* surfaceVertexCompactionCP,
                       ComputePipeline* surfaceVertexDensityCP,
                       ComputePipeline* surfaceVertexNormalCP,
                       MeshPipeline* fluidMeshPipeline)
    : Drawable(context, pipeline), 
      bilevelUniformGridCP(bilevelUniformGridCP), 
      surfaceBlockDetectionCP(surfaceBlockDetectionCP),
      surfaceCellDetectionCP(surfaceCellDetectionCP),
      surfaceVertexCompactionCP(surfaceVertexCompactionCP),
      surfaceVertexDensityCP(surfaceVertexDensityCP),
      surfaceVertexNormalCP(surfaceVertexNormalCP),
      fluidMeshPipeline(fluidMeshPipeline)
{
    constructScene();
}

// In this pipeline, drawing is done via a mesh shader
void FluidScene::draw(Camera* camera) {
    auto cmdList = fluidMeshPipeline->getCommandList();
    MeshShadingConstants meshShadingConstants = { camera->getViewProjMat(), gridConstants.gridDim, gridConstants.resolution, gridConstants.minBounds };

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
    cmdList->SetGraphicsRoot32BitConstants(5, 23, &meshShadingConstants, 0);

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

    transitionBuffers(cmdList, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE|D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
    resetBuffers(cmdList);
    transitionBuffers(cmdList, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    context->executeCommandList(fluidMeshPipeline->getCommandListID());
    context->signalAndWaitForFence(fence, fenceValue);

    context->resetCommandList(fluidMeshPipeline->getCommandListID());
}

float getRandomFloatInRange(float min, float max) {
    return min + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (max - min)));
}

void FluidScene::constructScene() {
    int blocksPerEdge = 16;
    int numParticlesPerCell = 8;
    int numParticles = numParticlesPerCell * blocksPerEdge * CELLS_PER_BLOCK_EDGE * blocksPerEdge * CELLS_PER_BLOCK_EDGE * blocksPerEdge * CELLS_PER_BLOCK_EDGE;
    numParticles -= numParticlesPerCell * blocksPerEdge * CELLS_PER_BLOCK_EDGE * blocksPerEdge * CELLS_PER_BLOCK_EDGE; // Skip the top level of cells
    gridConstants = { numParticles, {blocksPerEdge * CELLS_PER_BLOCK_EDGE, blocksPerEdge * CELLS_PER_BLOCK_EDGE, blocksPerEdge * CELLS_PER_BLOCK_EDGE}, {0.f, 0.f, 0.f}, 0.01f };

    // Populate position data. Place a particle per-cell in a 12x12x12 block of cells, each at a random position in a cell.
    // (Temporary, eventually, position data will come from simulation)
    for (int u = 0; u < numParticlesPerCell; ++u) {
        for (int i = 0; i < gridConstants.gridDim.x; ++i) {
            for (int j = 0; j < gridConstants.gridDim.y; ++j) {
                for (int k = 0; k < gridConstants.gridDim.z; ++k) {
                    // Skip i,j,k = top level of cells, that way the top level is empty and the second-to-top level becomes the "surface"
                    if (j == gridConstants.gridDim.z - 1) {
                        continue;
                    }

                    positions.push_back({ 
                        gridConstants.minBounds.x + gridConstants.resolution * i + getRandomFloatInRange(0.f, gridConstants.resolution),
                        gridConstants.minBounds.y + gridConstants.resolution * j + getRandomFloatInRange(0.f, gridConstants.resolution),
                        gridConstants.minBounds.z + gridConstants.resolution * k + getRandomFloatInRange(0.f, gridConstants.resolution) 
                    });
                }
            }
        }
    }

    positionBuffer = StructuredBuffer(positions.data(), gridConstants.numParticles, sizeof(XMFLOAT3));
    positionBuffer.passDataToGPU(*context, bilevelUniformGridCP->getCommandList(), bilevelUniformGridCP->getCommandListID());
    positionBuffer.createSRV(*context, bilevelUniformGridCP->getDescriptorHeap());

    // Create cells and blocks buffers
    int numCells = gridConstants.gridDim.x * gridConstants.gridDim.y * gridConstants.gridDim.z;
    int numBlocks = numCells / (CELLS_PER_BLOCK_EDGE * CELLS_PER_BLOCK_EDGE * CELLS_PER_BLOCK_EDGE);
    int numVerts = (gridConstants.gridDim.x + 1) * (gridConstants.gridDim.y + 1) * (gridConstants.gridDim.z + 1);
    
    std::vector<Cell> cells(numCells);
    std::vector<Block> blocks(numBlocks);
    std::vector<int> surfaceBlockIndices(numBlocks, 0);
    XMUINT3 dipatchCPU = { 0, 1, 1 };
    std::vector<int> surfaceVertices(numVerts, 0);
    std::vector<int> surfaceVertexIndices(numVerts, 0);
    std::vector<float> surfaceVertexDensities(numVerts, 1.f); // initialize density to 1 (and reset it to 1). Just has to be above ISOVALUE
    std::vector<XMFLOAT3> surfaceVertexNormals(numVerts, { 0.f, 0.f, 0.f });

    // Use the descriptor heap for the bilevelUniformGridCP for pretty much everything. Simplifies sharing resources
    blocksBuffer = StructuredBuffer(blocks.data(), numBlocks, sizeof(Block));
    blocksBuffer.passDataToGPU(*context, bilevelUniformGridCP->getCommandList(), bilevelUniformGridCP->getCommandListID());
    blocksBuffer.createUAV(*context, bilevelUniformGridCP->getDescriptorHeap());
    blocksBuffer.createSRV(*context, bilevelUniformGridCP->getDescriptorHeap());

    cellsBuffer = StructuredBuffer(cells.data(), numCells, sizeof(Cell));
    cellsBuffer.passDataToGPU(*context, bilevelUniformGridCP->getCommandList(), bilevelUniformGridCP->getCommandListID());
    cellsBuffer.createUAV(*context, bilevelUniformGridCP->getDescriptorHeap());
    cellsBuffer.createSRV(*context, bilevelUniformGridCP->getDescriptorHeap());

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
    
    // Blank buffers (for resetting their non-blank counterparts)
    blankCellsBuffer = StructuredBuffer(cells.data(), numCells, sizeof(Cell));
    blankCellsBuffer.passDataToGPU(*context, bilevelUniformGridCP->getCommandList(), bilevelUniformGridCP->getCommandListID());

    blankBlocksBuffer = StructuredBuffer(blocks.data(), numBlocks, sizeof(Block));
    blankBlocksBuffer.passDataToGPU(*context, bilevelUniformGridCP->getCommandList(), bilevelUniformGridCP->getCommandListID());

    blankSurfaceVerticesBuffer = StructuredBuffer(surfaceVertices.data(), numVerts, sizeof(unsigned int));
    blankSurfaceVerticesBuffer.passDataToGPU(*context, bilevelUniformGridCP->getCommandList(), bilevelUniformGridCP->getCommandListID());

    blankSurfaceVertDensityBuffer = StructuredBuffer(surfaceVertexDensities.data(), numVerts, sizeof(float));
    blankSurfaceVertDensityBuffer.passDataToGPU(*context, bilevelUniformGridCP->getCommandList(), bilevelUniformGridCP->getCommandListID());

    blankSurfaceVertexNormalBuffer = StructuredBuffer(surfaceVertexNormals.data(), numVerts, sizeof(XMFLOAT3));
    blankSurfaceVertexNormalBuffer.passDataToGPU(*context, bilevelUniformGridCP->getCommandList(), bilevelUniformGridCP->getCommandListID());

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

void FluidScene::compute() {
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
    cmdList->SetComputeRootDescriptorTable(0, positionBuffer.getSRVGPUDescriptorHandle());
    cmdList->SetComputeRootDescriptorTable(1, cellsBuffer.getUAVGPUDescriptorHandle());
    cmdList->SetComputeRootDescriptorTable(2, blocksBuffer.getUAVGPUDescriptorHandle());
    cmdList->SetComputeRoot32BitConstants(3, 8, &gridConstants, 0);

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
    D3D12_RESOURCE_BARRIER cellsBufferBarrier = CD3DX12_RESOURCE_BARRIER::Transition(
        cellsBuffer.getBuffer(),
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
    );

    D3D12_RESOURCE_BARRIER barriers[3] = { surfaceBlockIndicesBufferBarrier, surfaceBlockDispatchBarrier, cellsBufferBarrier };
    cmdList->ResourceBarrier(3, barriers);

    // Set compute root descriptor table
    cmdList->SetComputeRootDescriptorTable(0, surfaceBlockIndicesBuffer.getSRVGPUDescriptorHandle());
    cmdList->SetComputeRootDescriptorTable(1, cellsBuffer.getSRVGPUDescriptorHandle());
    cmdList->SetComputeRootDescriptorTable(2, surfaceVerticesBuffer.getUAVGPUDescriptorHandle());
    cmdList->SetComputeRootShaderResourceView(3, surfaceBlockDispatch.getGPUVirtualAddress());
    cmdList->SetComputeRootUnorderedAccessView(4, surfaceHalfBlockDispatch.getGPUVirtualAddress());
    cmdList->SetComputeRoot32BitConstants(5, 3, &gridConstants.gridDim, 0);

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


    // Transition surfaceBlockDispatch to an SRV 
    D3D12_RESOURCE_BARRIER surfaceBlockDispatchBarrier = CD3DX12_RESOURCE_BARRIER::Transition(
        surfaceBlockDispatch.getBuffer(),
        D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
    );

    // Transition surfaceVerticesBuffer back to SRV 
    D3D12_RESOURCE_BARRIER surfaceVerticesBufferBarrier = CD3DX12_RESOURCE_BARRIER::Transition(
        surfaceVerticesBuffer.getBuffer(),
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE 
    );

    D3D12_RESOURCE_BARRIER barriers[2] = { surfaceBlockDispatchBarrier, surfaceVerticesBufferBarrier };
    cmdList->ResourceBarrier(2, barriers);

    // Set compute root descriptor table
    cmdList->SetComputeRootDescriptorTable(0, surfaceVerticesBuffer.getSRVGPUDescriptorHandle());
    //make sure this should be block and SRV
    cmdList->SetComputeRootDescriptorTable(1, surfaceBlockDispatch.getSRVGPUDescriptorHandle());
    cmdList->SetComputeRootDescriptorTable(2, surfaceVertexIndicesBuffer.getUAVGPUDescriptorHandle());

    cmdList->SetComputeRootUnorderedAccessView(3, surfaceVertDensityDispatch.getGPUVirtualAddress());

    // Transition surfaceBlockDispatch to indirect argument buffer
    D3D12_RESOURCE_BARRIER surfaceBlockDispatchBarrier2 = CD3DX12_RESOURCE_BARRIER::Transition(
        surfaceBlockDispatch.getBuffer(),
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT
    );

    cmdList->ResourceBarrier(1, &surfaceBlockDispatchBarrier2);

    // Dispatch
    cmdList->ExecuteIndirect(commandSignature, 1, surfaceBlockDispatch.getBuffer(), 0, nullptr, 0);

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
    cmdList->SetComputeRootDescriptorTable(0, positionBuffer.getSRVGPUDescriptorHandle());
    cmdList->SetComputeRootDescriptorTable(1, cellsBuffer.getSRVGPUDescriptorHandle());
    cmdList->SetComputeRootDescriptorTable(2, surfaceVertexIndicesBuffer.getSRVGPUDescriptorHandle());
    cmdList->SetComputeRootShaderResourceView(3, surfaceVertDensityDispatch.getGPUVirtualAddress());
    cmdList->SetComputeRootUnorderedAccessView(4, surfaceBlockDispatch.getGPUVirtualAddress());
    cmdList->SetComputeRootDescriptorTable(5, surfaceVertDensityBuffer.getUAVGPUDescriptorHandle());
    cmdList->SetComputeRoot32BitConstants(6, 8, &gridConstants, 0);


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
    cmdList->SetComputeRoot32BitConstants(4, 8, &gridConstants, 0);

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
    positionBuffer.releaseResources();
    cellsBuffer.releaseResources();
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
    D3D12_RESOURCE_BARRIER cellsBufferBarrier = CD3DX12_RESOURCE_BARRIER::Transition(
        cellsBuffer.getBuffer(),
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

    D3D12_RESOURCE_BARRIER barriers[10] = { 
        cellsBufferBarrier,
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

    cmdList->ResourceBarrier(10, barriers);
}

// This is a GPU->GPU copy, so it's fast, but it requires allocating extra memory for the blank buffers.
// We could also use a compute shader to reset the buffers in the future.
void FluidScene::resetBuffers(ID3D12GraphicsCommandList6* cmdList) {
    cmdList->CopyBufferRegion(cellsBuffer.getBuffer(), 0, blankCellsBuffer.getBuffer(), 0, blankCellsBuffer.getElementSize() * blankCellsBuffer.getNumElements());
    cmdList->CopyBufferRegion(blocksBuffer.getBuffer(), 0, blankBlocksBuffer.getBuffer(), 0, blankBlocksBuffer.getElementSize() * blankBlocksBuffer.getNumElements());
    cmdList->CopyBufferRegion(surfaceVerticesBuffer.getBuffer(), 0, blankSurfaceVerticesBuffer.getBuffer(), 0, blankSurfaceVerticesBuffer.getElementSize() * blankSurfaceVerticesBuffer.getNumElements());
    cmdList->CopyBufferRegion(surfaceVertexIndicesBuffer.getBuffer(), 0, blankSurfaceVerticesBuffer.getBuffer(), 0, blankSurfaceVerticesBuffer.getElementSize() * blankSurfaceVerticesBuffer.getNumElements());
    cmdList->CopyBufferRegion(surfaceVertDensityBuffer.getBuffer(), 0, blankSurfaceVertDensityBuffer.getBuffer(), 0, blankSurfaceVertDensityBuffer.getElementSize() * blankSurfaceVertDensityBuffer.getNumElements());
}