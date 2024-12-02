#include "PBMPMScene.h"

PBMPMScene::PBMPMScene(DXContext* context, RenderPipeline* pipeline, unsigned int instances)
	: Drawable(context, pipeline), context(context), renderPipeline(pipeline), instanceCount(instances),
	modelMat(XMMatrixIdentity()),
	g2p2gPipeline("g2p2gRootSignature.cso", "g2p2gComputeShader.cso", *context, CommandListID::PBMPM_G2P2G_COMPUTE_ID,
		D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 30, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE),
	bukkitCountPipeline("bukkitCountRootSignature.cso", "bukkitCountComputeShader.cso", *context, CommandListID::PBMPM_BUKKITCOUNT_COMPUTE_ID,
		D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE),
	bukkitAllocatePipeline("bukkitAllocateRootSignature.cso", "bukkitAllocateComputeShader.cso", *context, CommandListID::PBMPM_BUKKITALLOCATE_COMPUTE_ID,
		D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE),
	bukkitInsertPipeline("bukkitInsertRootSignature.cso", "bukkitInsertComputeShader.cso", *context, CommandListID::PBMPM_BUKKITINSERT_COMPUTE_ID,
		D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE),
	bufferClearPipeline("bufferClearRootSignature.cso", "bufferClearComputeShader.cso", *context, CommandListID::PBMPM_BUFFERCLEAR_COMPUTE_ID,
		D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE)

{
	
	constructScene();
}

void PBMPMScene::createBukkitSystem() {
	int bukkitCountX = std::ceil(constants.gridSize.x / BukkitSize);
	int bukkitCountY = std::ceil(constants.gridSize.y / BukkitSize);
	int bukkitCountZ = std::ceil(constants.gridSize.z / BukkitSize);

	std::vector<int> count;
	count.resize(bukkitCountX * bukkitCountY * bukkitCountZ);
	bukkitSystem.countBuffer = StructuredBuffer(count.data(), count.size(), sizeof(int));

	std::vector<int> count2;
	count2.resize(bukkitCountX * bukkitCountY * bukkitCountZ);
	bukkitSystem.countBuffer2 = StructuredBuffer(count2.data(), count2.size(), sizeof(int));

	std::vector<int> particleData;
	particleData.resize(maxParticles);
	bukkitSystem.particleData = StructuredBuffer(particleData.data(), particleData.size(), sizeof(int));

	std::vector<BukkitThreadData> threadData;
	threadData.resize(40 * bukkitCountX * bukkitCountY * bukkitCountZ); //idk why this is 40
	bukkitSystem.threadData = StructuredBuffer(threadData.data(), threadData.size(), sizeof(BukkitThreadData));

	XMUINT4 allocator = { 0, 0, 0, 0 };
	bukkitSystem.particleAllocator = StructuredBuffer(&allocator, 1, sizeof(XMUINT4));

	std::vector<int> indexStart;
	indexStart.resize(bukkitCountX * bukkitCountY * bukkitCountZ);
	bukkitSystem.indexStart = StructuredBuffer(indexStart.data(), indexStart.size(), sizeof(int));

	XMUINT4 dispatch = { 0, 1, 1, 0 };
	bukkitSystem.dispatch = StructuredBuffer(&dispatch, 1, sizeof(XMUINT4));

	XMUINT4 blankDispatch = { 0, 1, 1, 0 };
	bukkitSystem.blankDispatch = StructuredBuffer(&blankDispatch, 1, sizeof(XMUINT4));

	bukkitSystem.countX = bukkitCountX;
	bukkitSystem.countY = bukkitCountY;
	bukkitSystem.countZ = bukkitCountZ;
	bukkitSystem.count = bukkitCountX * bukkitCountY * bukkitCountZ;
	bukkitSystem.countBuffer.passDataToGPU(*context, bukkitCountPipeline.getCommandList(), bukkitCountPipeline.getCommandListID());
	bukkitSystem.countBuffer2.passDataToGPU(*context, bukkitInsertPipeline.getCommandList(), bukkitInsertPipeline.getCommandListID());
	bukkitSystem.particleData.passDataToGPU(*context, bukkitInsertPipeline.getCommandList(), bukkitInsertPipeline.getCommandListID());
	bukkitSystem.threadData.passDataToGPU(*context, bukkitAllocatePipeline.getCommandList(), bukkitAllocatePipeline.getCommandListID());
	bukkitSystem.particleAllocator.passDataToGPU(*context, bukkitAllocatePipeline.getCommandList(), bukkitAllocatePipeline.getCommandListID());
	bukkitSystem.indexStart.passDataToGPU(*context, bukkitAllocatePipeline.getCommandList(), bukkitAllocatePipeline.getCommandListID());
	bukkitSystem.dispatch.passDataToGPU(*context, bukkitAllocatePipeline.getCommandList(), bukkitAllocatePipeline.getCommandListID());

	// Create UAV's for each buffer
	bukkitSystem.countBuffer.createUAV(*context, g2p2gPipeline.getDescriptorHeap());
	bukkitSystem.countBuffer2.createUAV(*context, g2p2gPipeline.getDescriptorHeap());
	bukkitSystem.particleData.createUAV(*context, g2p2gPipeline.getDescriptorHeap());
	bukkitSystem.threadData.createUAV(*context, g2p2gPipeline.getDescriptorHeap());
	bukkitSystem.particleAllocator.createUAV(*context, g2p2gPipeline.getDescriptorHeap());
	bukkitSystem.indexStart.createUAV(*context, g2p2gPipeline.getDescriptorHeap());
	bukkitSystem.dispatch.createUAV(*context, g2p2gPipeline.getDescriptorHeap());

	// Create SRV's for each buffer
	bukkitSystem.countBuffer.createSRV(*context, g2p2gPipeline.getDescriptorHeap());
	bukkitSystem.countBuffer2.createSRV(*context, g2p2gPipeline.getDescriptorHeap());
	bukkitSystem.particleData.createSRV(*context, g2p2gPipeline.getDescriptorHeap());
	bukkitSystem.threadData.createSRV(*context, g2p2gPipeline.getDescriptorHeap());
	bukkitSystem.particleAllocator.createSRV(*context, g2p2gPipeline.getDescriptorHeap());
	bukkitSystem.indexStart.createSRV(*context, g2p2gPipeline.getDescriptorHeap());
	bukkitSystem.dispatch.createSRV(*context, g2p2gPipeline.getDescriptorHeap());

	bukkitSystem.blankDispatch.passCBVDataToGPU(*context, bukkitCountPipeline.getDescriptorHeap());
}

void PBMPMScene::updateSimUniforms(unsigned int iteration) {
	// DO MOUSE UPDATING HERE
	constants.simFrame = substepIndex;
	constants.bukkitCount = bukkitSystem.count;
	constants.bukkitCountX = bukkitSystem.countX;
	constants.bukkitCountY = bukkitSystem.countY;
	constants.bukkitCountZ = bukkitSystem.countZ;
	constants.iteration = iteration;
}

void PBMPMScene::resetBuffers(bool resetGrids) {
	//clear buffers (Make sure each one is a UAV)
	constexpr UINT THREAD_GROUP_SIZE = 256;

	// Bind the PSO and Root Signature
	bufferClearPipeline.getCommandList()->SetPipelineState(bufferClearPipeline.getPSO());
	bufferClearPipeline.getCommandList()->SetComputeRootSignature(bufferClearPipeline.getRootSignature());

	// Bind the descriptor heap
	ID3D12DescriptorHeap* descriptorHeaps[] = { g2p2gPipeline.getDescriptorHeap()->Get() };
	bufferClearPipeline.getCommandList()->SetDescriptorHeaps(_countof(descriptorHeaps), descriptorHeaps);

	// Reset CountBuffer:
	UINT countSize = bukkitSystem.count; // The total number of elements in the buffer
	bufferClearPipeline.getCommandList()->SetComputeRoot32BitConstants(0, 1, &countSize, 0);
	bufferClearPipeline.getCommandList()->SetComputeRootDescriptorTable(1, bukkitSystem.countBuffer.getUAVGPUDescriptorHandle());
	bufferClearPipeline.getCommandList()->Dispatch((countSize + THREAD_GROUP_SIZE - 1) / THREAD_GROUP_SIZE, 1, 1);

	// Reset CountBuffer2:
	bufferClearPipeline.getCommandList()->SetComputeRoot32BitConstants(0, 1, &countSize, 0);
	bufferClearPipeline.getCommandList()->SetComputeRootDescriptorTable(1, bukkitSystem.countBuffer2.getUAVGPUDescriptorHandle());
	bufferClearPipeline.getCommandList()->Dispatch((countSize + THREAD_GROUP_SIZE - 1) / THREAD_GROUP_SIZE, 1, 1);

	// Reset ParticleData:
	UINT particleDataSize = maxParticles; // The total number of elements in the buffer
	bufferClearPipeline.getCommandList()->SetComputeRoot32BitConstants(0, 1, &particleDataSize, 0);
	bufferClearPipeline.getCommandList()->SetComputeRootDescriptorTable(1, bukkitSystem.particleData.getUAVGPUDescriptorHandle());
	bufferClearPipeline.getCommandList()->Dispatch((particleDataSize + THREAD_GROUP_SIZE - 1) / THREAD_GROUP_SIZE, 1, 1);

	// Reset ThreadData:
	UINT threadDataSize = 40 * bukkitSystem.count; // The total number of elements in the buffer - this was also 40
	bufferClearPipeline.getCommandList()->SetComputeRoot32BitConstants(0, 1, &threadDataSize, 0);
	bufferClearPipeline.getCommandList()->SetComputeRootDescriptorTable(1, bukkitSystem.threadData.getUAVGPUDescriptorHandle());
	bufferClearPipeline.getCommandList()->Dispatch((threadDataSize + THREAD_GROUP_SIZE - 1) / THREAD_GROUP_SIZE, 1, 1);

	// Reset ParticleAllocator:
	UINT particleAllocatorSize = 4; // The total number of elements in the buffer
	bufferClearPipeline.getCommandList()->SetComputeRoot32BitConstants(0, 1, &particleAllocatorSize, 0);
	bufferClearPipeline.getCommandList()->SetComputeRootDescriptorTable(1, bukkitSystem.particleAllocator.getUAVGPUDescriptorHandle());
	bufferClearPipeline.getCommandList()->Dispatch((particleAllocatorSize + THREAD_GROUP_SIZE - 1) / THREAD_GROUP_SIZE, 1, 1);

	// Transition dispatch buffer to a copy destination
	D3D12_RESOURCE_BARRIER dispatchBarrier = CD3DX12_RESOURCE_BARRIER::Transition(bukkitSystem.dispatch.getBuffer(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
	bufferClearPipeline.getCommandList()->ResourceBarrier(1, &dispatchBarrier);

	// Copy blank dispatch to dispatch (reset dispatch)
	bukkitInsertPipeline.getCommandList()->CopyBufferRegion(bukkitSystem.dispatch.getBuffer(), 0, bukkitSystem.blankDispatch.getBuffer(), 0, sizeof(XMUINT4));

	// Transition dispatch buffer back to a non-pixel shader resource
	dispatchBarrier = CD3DX12_RESOURCE_BARRIER::Transition(bukkitSystem.dispatch.getBuffer(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	bufferClearPipeline.getCommandList()->ResourceBarrier(1, &dispatchBarrier);

	// Reset grid buffers
	if (resetGrids) {
		for (int i = 0; i < 3; i++) {
			UINT numGridInts = constants.gridSize.x * constants.gridSize.y * constants.gridSize.z * sizeof(UINT); // The total number of elements in the buffers
			bufferClearPipeline.getCommandList()->SetComputeRoot32BitConstants(0, 1, &numGridInts, 0);
			bufferClearPipeline.getCommandList()->SetComputeRootDescriptorTable(1, gridBuffers[i].getUAVGPUDescriptorHandle());
			bufferClearPipeline.getCommandList()->Dispatch((numGridInts + THREAD_GROUP_SIZE - 1) / THREAD_GROUP_SIZE, 1, 1);
		}
	}

	// execute
	context->executeCommandList(bufferClearPipeline.getCommandListID());
	context->executeCommandList(bukkitInsertPipeline.getCommandListID());

	// Use a fence to synchronize the completion of the command lists
	context->signalAndWaitForFence(fence, fenceValue);

	// Reset the command lists
	context->resetCommandList(bufferClearPipeline.getCommandListID());
	context->resetCommandList(bukkitInsertPipeline.getCommandListID());
}

void PBMPMScene::bukkitizeParticles() {
	
	// Reset Buffers, but not the grid
	resetBuffers(false);

	// Bind the PSO and Root Signature
	bukkitCountPipeline.getCommandList()->SetPipelineState(bukkitCountPipeline.getPSO());
	bukkitCountPipeline.getCommandList()->SetComputeRootSignature(bukkitCountPipeline.getRootSignature());

	// Bind the descriptor heap
	ID3D12DescriptorHeap* descriptorHeaps[] = { g2p2gPipeline.getDescriptorHeap()->Get() };
	bukkitCountPipeline.getCommandList()->SetDescriptorHeaps(_countof(descriptorHeaps), descriptorHeaps);

	// Transition particle buffer to srv
	D3D12_RESOURCE_BARRIER particleBufferBarrier = CD3DX12_RESOURCE_BARRIER::Transition(particleBuffer.getBuffer(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	bukkitCountPipeline.getCommandList()->ResourceBarrier(1, &particleBufferBarrier);

	// Properly set the Descriptors & Resource Transitions
	bukkitCountPipeline.getCommandList()->SetComputeRoot32BitConstants(0, 20, &constants, 0);
	bukkitCountPipeline.getCommandList()->SetComputeRootDescriptorTable(1, particleCount.getSRVGPUDescriptorHandle());
	bukkitCountPipeline.getCommandList()->SetComputeRootDescriptorTable(2, particleBuffer.getSRVGPUDescriptorHandle());
	bukkitCountPipeline.getCommandList()->SetComputeRootDescriptorTable(3, bukkitSystem.countBuffer.getUAVGPUDescriptorHandle());

	// Transition particleSimDispatch to indirect dispatch
	D3D12_RESOURCE_BARRIER particleSimDispatchBarrier = CD3DX12_RESOURCE_BARRIER::Transition(particleSimDispatch.getBuffer(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
	bukkitCountPipeline.getCommandList()->ResourceBarrier(1, &particleSimDispatchBarrier);

	//dispatch indirectly <3
	bukkitCountPipeline.getCommandList()->ExecuteIndirect(commandSignature, 1, particleSimDispatch.getBuffer(), 0, nullptr, 0);

	// Transition particleSimDispatch back to unordered access
	D3D12_RESOURCE_BARRIER particleSimDispatchBarrierEnd = CD3DX12_RESOURCE_BARRIER::Transition(particleSimDispatch.getBuffer(), D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
	bukkitCountPipeline.getCommandList()->ResourceBarrier(1, &particleSimDispatchBarrierEnd);

	// Transition particle buffer back to UAV
	D3D12_RESOURCE_BARRIER particleBufferBarrierEnd = CD3DX12_RESOURCE_BARRIER::Transition(particleBuffer.getBuffer(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
	bukkitCountPipeline.getCommandList()->ResourceBarrier(1, &particleBufferBarrierEnd);

	// execute
	context->executeCommandList(bukkitCountPipeline.getCommandListID());

	// Use a fence to synchronize the completion of the command lists
	context->signalAndWaitForFence(fence, fenceValue);

	// Reset the command lists
	context->resetCommandList(bukkitCountPipeline.getCommandListID());
	
	auto bukkitDispatchSizeX = std::floor((bukkitSystem.countX + GridDispatchSize - 1) / GridDispatchSize);
	auto bukkitDispatchSizeY = std::floor((bukkitSystem.countY + GridDispatchSize - 1) / GridDispatchSize);
	auto bukkitDispatchSizeZ = std::floor((bukkitSystem.countZ + GridDispatchSize - 1) / GridDispatchSize);

	// Bind the PSO and Root Signature
	bukkitAllocatePipeline.getCommandList()->SetPipelineState(bukkitAllocatePipeline.getPSO());
	bukkitAllocatePipeline.getCommandList()->SetComputeRootSignature(bukkitAllocatePipeline.getRootSignature());

	// Bind the descriptor heap
	bukkitAllocatePipeline.getCommandList()->SetDescriptorHeaps(_countof(descriptorHeaps), descriptorHeaps);

	// Transition bukkitCount to srv
	D3D12_RESOURCE_BARRIER bukkitCountBarrier = CD3DX12_RESOURCE_BARRIER::Transition(bukkitSystem.countBuffer.getBuffer(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	bukkitAllocatePipeline.getCommandList()->ResourceBarrier(1, &bukkitCountBarrier);

	// Properly set the Descriptors & Resource Transitions
	bukkitAllocatePipeline.getCommandList()->SetComputeRoot32BitConstants(0, 20, &constants, 0);
	bukkitAllocatePipeline.getCommandList()->SetComputeRootDescriptorTable(1, bukkitSystem.countBuffer.getSRVGPUDescriptorHandle());
	bukkitAllocatePipeline.getCommandList()->SetComputeRootDescriptorTable(2, bukkitSystem.threadData.getUAVGPUDescriptorHandle());

	//dispatch directly
	bukkitAllocatePipeline.getCommandList()->Dispatch(bukkitDispatchSizeX, bukkitDispatchSizeY, bukkitDispatchSizeZ);

	// Transition bukkitCount back to UAV
	D3D12_RESOURCE_BARRIER bukkitCountBarrierEnd = CD3DX12_RESOURCE_BARRIER::Transition(bukkitSystem.countBuffer.getBuffer(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
	bukkitAllocatePipeline.getCommandList()->ResourceBarrier(1, &bukkitCountBarrierEnd);

	// execute
	context->executeCommandList(bukkitAllocatePipeline.getCommandListID());

	// Use a fence to synchronize the completion of the command lists
	context->signalAndWaitForFence(fence, fenceValue);

	// Reset the command lists
	context->resetCommandList(bukkitAllocatePipeline.getCommandListID());

	//// Test CPU Data things
	//std::vector<int> particleAllocatorCPU;
	//particleAllocatorCPU.resize(4);
	//bukkitSystem.particleAllocator.copyDataFromGPU(*context, particleAllocatorCPU.data(), bukkitAllocatePipeline.getCommandList(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, bukkitAllocatePipeline.getCommandListID());

	//// Copy Thread Data
	//std::vector<BukkitThreadData> threadData;
	//threadData.resize(40 * bukkitSystem.count);
	//bukkitSystem.threadData.copyDataFromGPU(*context, threadData.data(), bukkitAllocatePipeline.getCommandList(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, bukkitAllocatePipeline.getCommandListID());

	//// Copy Index Start
	//std::vector<int> indexStart;
	//indexStart.resize(bukkitSystem.count);
	//bukkitSystem.indexStart.copyDataFromGPU(*context, indexStart.data(), bukkitAllocatePipeline.getCommandList(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, bukkitAllocatePipeline.getCommandListID());

	// Bind the PSO and Root Signature
	bukkitInsertPipeline.getCommandList()->SetPipelineState(bukkitInsertPipeline.getPSO());
	bukkitInsertPipeline.getCommandList()->SetComputeRootSignature(bukkitInsertPipeline.getRootSignature());

	// Bind the descriptor heap
	bukkitInsertPipeline.getCommandList()->SetDescriptorHeaps(_countof(descriptorHeaps), descriptorHeaps);

	// Transition particleBuffer, particleCount, and indexStart to SRV
	D3D12_RESOURCE_BARRIER particleBufferBarrier2 = CD3DX12_RESOURCE_BARRIER::Transition(particleBuffer.getBuffer(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	D3D12_RESOURCE_BARRIER particleCountBarrier = CD3DX12_RESOURCE_BARRIER::Transition(particleCount.getBuffer(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	D3D12_RESOURCE_BARRIER indexStartBarrier = CD3DX12_RESOURCE_BARRIER::Transition(bukkitSystem.indexStart.getBuffer(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	// Transition particle sim dispatch to indirect dispatch
	D3D12_RESOURCE_BARRIER particleSimDispatchBarrier2 = CD3DX12_RESOURCE_BARRIER::Transition(particleSimDispatch.getBuffer(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);

	// Create an array of barriers
	D3D12_RESOURCE_BARRIER barriers[4] = { particleBufferBarrier2, particleCountBarrier, indexStartBarrier, particleSimDispatchBarrier2};

	// Transition the resources
	bukkitInsertPipeline.getCommandList()->ResourceBarrier(3, barriers);

	// Properly set the Descriptors
	bukkitInsertPipeline.getCommandList()->SetComputeRoot32BitConstants(0, 20, &constants, 0);
	bukkitInsertPipeline.getCommandList()->SetComputeRootDescriptorTable(1, particleBuffer.getSRVGPUDescriptorHandle());
	bukkitInsertPipeline.getCommandList()->SetComputeRootDescriptorTable(2, bukkitSystem.countBuffer2.getUAVGPUDescriptorHandle());
	bukkitInsertPipeline.getCommandList()->SetComputeRootDescriptorTable(3, bukkitSystem.indexStart.getSRVGPUDescriptorHandle());

	// Dispatch indirectly again
	bukkitInsertPipeline.getCommandList()->ExecuteIndirect(commandSignature, 1, particleSimDispatch.getBuffer(), 0, nullptr, 0);

	// Transition particleBuffer, particleCount, and indexStart back to UAV
	D3D12_RESOURCE_BARRIER particleBufferBarrierEnd2 = CD3DX12_RESOURCE_BARRIER::Transition(particleBuffer.getBuffer(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
	D3D12_RESOURCE_BARRIER particleCountBarrierEnd = CD3DX12_RESOURCE_BARRIER::Transition(particleCount.getBuffer(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
	D3D12_RESOURCE_BARRIER indexStartBarrierEnd = CD3DX12_RESOURCE_BARRIER::Transition(bukkitSystem.indexStart.getBuffer(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
	// Transition particle sim dispatch back to unordered access
	D3D12_RESOURCE_BARRIER particleSimDispatchBarrierEnd2 = CD3DX12_RESOURCE_BARRIER::Transition(particleSimDispatch.getBuffer(), D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

	// Create an array of barriers
	D3D12_RESOURCE_BARRIER barriersEnd[4] = { particleBufferBarrierEnd2, particleCountBarrierEnd, indexStartBarrierEnd, particleSimDispatchBarrierEnd2 };

	// Transition the resources
	bukkitInsertPipeline.getCommandList()->ResourceBarrier(3, barriersEnd);

	// execute
	context->executeCommandList(bukkitInsertPipeline.getCommandListID());

	// Use a fence to synchronize the completion of the command lists
	context->signalAndWaitForFence(fence, fenceValue);

	// Reset the command lists
	context->resetCommandList(bukkitInsertPipeline.getCommandListID());

	// Copy from Count Buffers 2
	//std::vector<int> count;
	//count.resize(bukkitSystem.count);
	//bukkitSystem.countBuffer2.copyDataFromGPU(*context, count.data(), bukkitInsertPipeline.getCommandList(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, bukkitInsertPipeline.getCommandListID());

	// Copy from Particle Data
	//std::vector<int> particleData;
	//particleData.resize(maxParticles);
	//bukkitSystem.particleData.copyDataFromGPU(*context, particleData.data(), bukkitInsertPipeline.getCommandList(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, bukkitInsertPipeline.getCommandListID());

}

void PBMPMScene::constructScene() {
	auto computeId = g2p2gPipeline.getCommandListID();

	// Create Constant Data
	constants = { {512, 512, 512}, 0.01, 2.5, 1.5, 0.01,
		(unsigned int)std::ceil(std::pow(10, 7)),
		1, 4, 30, 0, 0,  0, 0, 0, 0, 0, 5, 0.9 };

	float radius = 0.005;
	float spacing = radius * 2.1;

	int particlesPerRow = (int)sqrt(instanceCount);
	int particlesPerCol = (instanceCount - 1) / particlesPerRow + 1;

	float gridSize = 512.0f;
	float guardianOffset = GuardianSize + 1.0f;
	float usableGridSize = gridSize - 2 * guardianOffset;
	float scalingFactor = usableGridSize / particlesPerRow;


	std::vector<PBMPMParticle> particles;
	particles.resize(maxParticles);
	// Uniform for each particle for now
	const float density = 1.f;
	const float volume = 1.f / float(constants.particlesPerCellAxis * constants.particlesPerCellAxis * constants.particlesPerCellAxis);
	// Create initial particle data
	for (int i = 0; i < instanceCount; ++i) {

		float x = ((i % particlesPerRow) * spacing * scalingFactor) + guardianOffset;
		float y = ((i / particlesPerRow) * spacing * scalingFactor) + guardianOffset;
		float z = guardianOffset;
		XMFLOAT3 position = { x, y, z };

		/*XMFLOAT3 position ={ (((i % particlesPerRow) * spacing - (particlesPerRow - 1) * spacing / 2.f) + 0.4f) * 500,
							  (((i / particlesPerRow) * spacing - (particlesPerCol - 1) * spacing / 2.f) + 0.4f) * 500, 0.f};*/
		particles[i] = {position, 1.0, {0.f, 0.f, 0.f}, density * volume, 0, volume, 0.0, 1.0, 1.0,
						{ 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0 },
						{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } };

	}

	particleBuffer = StructuredBuffer(particles.data(), particles.size(), sizeof(PBMPMParticle));
	
	std::vector<int> freeIndices;
	freeIndices.resize(1 + maxParticles); //maybe four maybe one idk

	XMUINT4 count = { 0, 0, 0, 0 };

	// Add particles
	for (int i = 0; i < instanceCount; i++) {
		freeIndices[0]--;
		if (freeIndices[0] < 0) {
			count.x++;
		}
	}

	particleCount = StructuredBuffer(&count, 1, sizeof(XMUINT4));
	particleFreeIndicesBuffer = StructuredBuffer(freeIndices.data(), freeIndices.size(), sizeof(int));
	
	// Set it based on instance size
	XMUINT4 simDispatch = { (instanceCount + ParticleDispatchSize - 1) / ParticleDispatchSize, 1, 1, 0};
	particleSimDispatch = StructuredBuffer(&simDispatch, 1, sizeof(XMUINT4));

	// Pass Structured Buffers to Compute Pipeline

	particleBuffer.passDataToGPU(*context, g2p2gPipeline.getCommandList(), computeId);
	particleFreeIndicesBuffer.passDataToGPU(*context, g2p2gPipeline.getCommandList(), computeId);
	particleCount.passDataToGPU(*context, g2p2gPipeline.getCommandList(), computeId);
	particleSimDispatch.passDataToGPU(*context, g2p2gPipeline.getCommandList(), computeId);

	// Create UAV's for each buffer
	particleBuffer.createUAV(*context, g2p2gPipeline.getDescriptorHeap());
	particleFreeIndicesBuffer.createUAV(*context, g2p2gPipeline.getDescriptorHeap());
	particleCount.createUAV(*context, g2p2gPipeline.getDescriptorHeap());
	particleSimDispatch.createUAV(*context, g2p2gPipeline.getDescriptorHeap());

	// Create SRV's for particleBuffer & particleCount
	particleBuffer.createSRV(*context, g2p2gPipeline.getDescriptorHeap());
	particleCount.createSRV(*context, g2p2gPipeline.getDescriptorHeap());

	//buffers updated per frame
	bukkitSystem = BukkitSystem{};
	createBukkitSystem();

	std::vector<int> gridBufferData;
	gridBufferData.resize(constants.gridSize.x * constants.gridSize.y * constants.gridSize.z * 5); //LOOK : 4 or 5?

	for (int i = 0; i < 3; i++) {
		gridBuffers[i] = StructuredBuffer(gridBufferData.data(), gridBufferData.size(), sizeof(int));
		gridBuffers[i].passDataToGPU(*context, g2p2gPipeline.getCommandList(), g2p2gPipeline.getCommandListID());
		gridBuffers[i].createUAV(*context, g2p2gPipeline.getDescriptorHeap());
	}
	// Separate UAV and SRV creation on the descriptor heap to allow contigious access
	gridBuffers[0].createSRV(*context, g2p2gPipeline.getDescriptorHeap());
	gridBuffers[1].createSRV(*context, g2p2gPipeline.getDescriptorHeap());
	gridBuffers[2].createSRV(*context, g2p2gPipeline.getDescriptorHeap());

	// Create Vertex & Index Buffer
	auto sphereData = generateSphere(radius, 16, 16);
	indexCount = (unsigned int)sphereData.second.size();

	vertexBuffer = VertexBuffer(sphereData.first, (UINT)(sphereData.first.size() * sizeof(XMFLOAT3)), (UINT)sizeof(XMFLOAT3));
	vbv = vertexBuffer.passVertexDataToGPU(*context, renderPipeline->getCommandList());

	indexBuffer = IndexBuffer(sphereData.second, (UINT)(sphereData.second.size() * sizeof(unsigned int)));
	ibv = indexBuffer.passIndexDataToGPU(*context, renderPipeline->getCommandList());

	//Transition both buffers to their usable states
	D3D12_RESOURCE_BARRIER barriers[2] = {};

	// Vertex buffer barrier
	barriers[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
	barriers[0].Transition.pResource = vertexBuffer.getVertexBuffer().Get();
	barriers[0].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
	barriers[0].Transition.StateAfter = D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER;
	barriers[0].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

	// Index buffer barrier
	barriers[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
	barriers[1].Transition.pResource = indexBuffer.getIndexBuffer().Get();
	barriers[1].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
	barriers[1].Transition.StateAfter = D3D12_RESOURCE_STATE_INDEX_BUFFER;
	barriers[1].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

	renderPipeline->getCommandList()->ResourceBarrier(2, barriers);

	renderPipeline->createPSOD();
	renderPipeline->createPipelineState(context->getDevice());

	// Execute and reset render pipeline command list
	context->executeCommandList(renderPipeline->getCommandListID());
	context->resetCommandList(renderPipeline->getCommandListID());

	// Create Fence
	context->getDevice()->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));

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
}

void PBMPMScene::compute() {

	int bufferIdx = 0;
	
	resetBuffers(true);

	// Could be 20?
	int substepCount = 20;
	for (int substepIdx = 0; substepIdx < substepCount; substepIdx++) {

		// Update simulation uniforms
		constants.iteration = 0;
		updateSimUniforms(substepIdx);
		
		for (int iterationIdx = 0; iterationIdx < constants.iterationCount; iterationIdx++) {
			constants.iteration = iterationIdx;

			updateSimUniforms(substepIdx);

			StructuredBuffer* currentGrid = &gridBuffers[bufferIdx];
			StructuredBuffer* nextGrid = &gridBuffers[(bufferIdx + 1) % 3];
			StructuredBuffer* nextNextGrid = &gridBuffers[(bufferIdx + 2) % 3];
			bufferIdx = (bufferIdx + 1) % 3;

			auto cmdList = g2p2gPipeline.getCommandList();

			D3D12_RESOURCE_BARRIER barriers[] = {
			CD3DX12_RESOURCE_BARRIER::Transition(bukkitSystem.particleData.getBuffer(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
			CD3DX12_RESOURCE_BARRIER::Transition(bukkitSystem.threadData.getBuffer(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)
			};
			cmdList->ResourceBarrier(_countof(barriers), barriers);

			auto currGridBarrier = CD3DX12_RESOURCE_BARRIER::Transition(currentGrid->getBuffer(),
				D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
			cmdList->ResourceBarrier(1, &currGridBarrier);

			cmdList->SetPipelineState(g2p2gPipeline.getPSO());
			cmdList->SetComputeRootSignature(g2p2gPipeline.getRootSignature());

			g2p2gPipeline.getCommandList()->SetComputeRoot32BitConstants(0, 20, &constants, 0);

			ID3D12DescriptorHeap* computeDescriptorHeaps[] = { g2p2gPipeline.getDescriptorHeap()->Get() };
			cmdList->SetDescriptorHeaps(_countof(computeDescriptorHeaps), computeDescriptorHeaps);

			cmdList->SetComputeRootDescriptorTable(1, particleBuffer.getUAVGPUDescriptorHandle());
			cmdList->SetComputeRootDescriptorTable(2, bukkitSystem.particleData.getSRVGPUDescriptorHandle());
			cmdList->SetComputeRootDescriptorTable(3, currentGrid->getSRVGPUDescriptorHandle());
			cmdList->SetComputeRootDescriptorTable(4, nextGrid->getUAVGPUDescriptorHandle());
			cmdList->SetComputeRootDescriptorTable(5, nextNextGrid->getUAVGPUDescriptorHandle());

			// Transition dispatch buffer to an indirect argument
			auto dispatchBarrier = CD3DX12_RESOURCE_BARRIER::Transition(bukkitSystem.dispatch.getBuffer(),
				D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
			cmdList->ResourceBarrier(1, &dispatchBarrier);

			// Indirect Dispatch
			cmdList->ExecuteIndirect(commandSignature, 1, bukkitSystem.dispatch.getBuffer(), 0, nullptr, 0);

			// Transition dispatch buffer back to a UAV
			dispatchBarrier = CD3DX12_RESOURCE_BARRIER::Transition(bukkitSystem.dispatch.getBuffer(),
				D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
			cmdList->ResourceBarrier(1, &dispatchBarrier);

			// Transition currentGrid to UAV
			currGridBarrier = CD3DX12_RESOURCE_BARRIER::Transition(currentGrid->getBuffer(),
				D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
			cmdList->ResourceBarrier(1, &currGridBarrier);

			// Transition particleData and threadData to UAV
			D3D12_RESOURCE_BARRIER endBarriers[] = {
				CD3DX12_RESOURCE_BARRIER::Transition(bukkitSystem.particleData.getBuffer(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
				CD3DX12_RESOURCE_BARRIER::Transition(bukkitSystem.threadData.getBuffer(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
			};
			cmdList->ResourceBarrier(_countof(endBarriers), endBarriers);

			// Execute command list
			context->executeCommandList(g2p2gPipeline.getCommandListID());
			context->signalAndWaitForFence(fence, fenceValue);

			// Reinitialize command list
			context->resetCommandList(g2p2gPipeline.getCommandListID());
		}

		// TODO: Add Emission function
		bukkitizeParticles();

		substepIndex++;
	}
}

void PBMPMScene::draw(Camera* cam) {

	auto cmdList = renderPipeline->getCommandList();

	// IA
	cmdList->IASetVertexBuffers(0, 1, &vbv);
	cmdList->IASetIndexBuffer(&ibv);
	cmdList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

	// PSO
	cmdList->SetPipelineState(renderPipeline->getPSO());
	cmdList->SetGraphicsRootSignature(renderPipeline->getRootSignature());

	// Transition particle buffer to SRV
	auto srvBarrier = CD3DX12_RESOURCE_BARRIER::Transition(particleBuffer.getBuffer(),
		D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
	cmdList->ResourceBarrier(1, &srvBarrier);

	ID3D12DescriptorHeap* descriptorHeaps[] = { g2p2gPipeline.getDescriptorHeap()->Get()};
	cmdList->SetDescriptorHeaps(_countof(descriptorHeaps), descriptorHeaps);
	auto viewMat = cam->getViewMat();
	auto projMat = cam->getProjMat();
	cmdList->SetGraphicsRoot32BitConstants(0, 16, &viewMat, 0);
	cmdList->SetGraphicsRoot32BitConstants(0, 16, &projMat, 16);
	cmdList->SetGraphicsRoot32BitConstants(0, 16, &modelMat, 32);
	cmdList->SetGraphicsRootDescriptorTable(1, particleBuffer.getSRVGPUDescriptorHandle()); // Descriptor table slot 1 for position SRV

	// Draw
	cmdList->DrawIndexedInstanced(indexCount, instanceCount, 0, 0, 0);

	srvBarrier = CD3DX12_RESOURCE_BARRIER::Transition(particleBuffer.getBuffer(),
		D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
	cmdList->ResourceBarrier(1, &srvBarrier);
}

void PBMPMScene::releaseResources() {
	vertexBuffer.releaseResources();
	indexBuffer.releaseResources();

	g2p2gPipeline.releaseResources();
	renderPipeline->releaseResources();
	bukkitAllocatePipeline.releaseResources();
	bukkitCountPipeline.releaseResources();
	bukkitInsertPipeline.releaseResources();

	particleBuffer.releaseResources();
	particleFreeIndicesBuffer.releaseResources();
	particleCount.releaseResources();
	particleSimDispatch.releaseResources();
	for (int i = 0; i < 3; i++) {
		gridBuffers[i].releaseResources();
	}

	bukkitSystem.countBuffer.releaseResources();
	bukkitSystem.countBuffer2.releaseResources();
	bukkitSystem.particleData.releaseResources();
	bukkitSystem.threadData.releaseResources();
	bukkitSystem.dispatch.releaseResources();
	bukkitSystem.blankDispatch.releaseResources();
	bukkitSystem.particleAllocator.releaseResources();
	bukkitSystem.indexStart.releaseResources();

	commandSignature->Release();
}

void PBMPMScene::updateConstants(PBMPMConstants& newConstants) {
	constants.gravityStrength = newConstants.gravityStrength;
	constants.liquidRelaxation = newConstants.liquidRelaxation;
	constants.liquidViscosity = newConstants.liquidViscosity;
	constants.fixedPointMultiplier = newConstants.fixedPointMultiplier;
	constants.useGridVolumeForLiquid = newConstants.useGridVolumeForLiquid;
	constants.particlesPerCellAxis = newConstants.particlesPerCellAxis;
	constants.frictionAngle = newConstants.frictionAngle;
	constants.borderFriction = newConstants.borderFriction;
}

bool PBMPMScene::constantsEqual(PBMPMConstants& one, PBMPMConstants& two) {
	return one.gravityStrength == two.gravityStrength &&
		one.liquidRelaxation == two.liquidRelaxation &&
		one.liquidViscosity == two.liquidViscosity &&
		one.fixedPointMultiplier == two.fixedPointMultiplier &&
		one.useGridVolumeForLiquid == two.useGridVolumeForLiquid &&
		one.particlesPerCellAxis == two.particlesPerCellAxis &&
		one.frictionAngle == two.frictionAngle &&
		one.borderFriction == two.borderFriction;
}