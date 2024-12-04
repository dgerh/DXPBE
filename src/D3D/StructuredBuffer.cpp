#include "StructuredBuffer.h"


StructuredBuffer::StructuredBuffer(const void* inputData, unsigned int numEle, UINT eleSize)
	: data(inputData), numElements(numEle), elementSize(eleSize)
{
}

void StructuredBuffer::findFreeHandle(DescriptorHeap* dh, CD3DX12_CPU_DESCRIPTOR_HANDLE& cpuHandle, CD3DX12_GPU_DESCRIPTOR_HANDLE& gpuHandle) {
	unsigned int index = dh->GetNextAvailableIndex();
    cpuHandle = dh->GetCPUHandleAt(index);
	gpuHandle = dh->GetGPUHandleAt(index);
}

CD3DX12_CPU_DESCRIPTOR_HANDLE StructuredBuffer::getUAVCPUDescriptorHandle()
{
	return UAVcpuHandle;
}

CD3DX12_GPU_DESCRIPTOR_HANDLE StructuredBuffer::getUAVGPUDescriptorHandle()
{
	return UAVgpuHandle;
}

CD3DX12_CPU_DESCRIPTOR_HANDLE StructuredBuffer::getSRVCPUDescriptorHandle()
{
    return SRVcpuHandle;
}

CD3DX12_GPU_DESCRIPTOR_HANDLE StructuredBuffer::getSRVGPUDescriptorHandle()
{
    return SRVgpuHandle;
}

CD3DX12_CPU_DESCRIPTOR_HANDLE StructuredBuffer::getCBVCPUDescriptorHandle()
{
	return CBVcpuHandle;
}

CD3DX12_GPU_DESCRIPTOR_HANDLE StructuredBuffer::getCBVGPUDescriptorHandle()
{
	return CBVgpuHandle;
}

D3D12_GPU_VIRTUAL_ADDRESS StructuredBuffer::getGPUVirtualAddress()
{
	return buffer->GetGPUVirtualAddress();
}

void StructuredBuffer::passCBVDataToGPU(DXContext& context, DescriptorHeap* dh) {

	if (isCBV) {
		throw std::runtime_error("CBV already created.");
    }
	else if (isUAV || isSRV) {
		throw std::runtime_error("Cannot create CBV after creating UAV or SRV.");
	}

    findFreeHandle(dh, CBVcpuHandle, CBVgpuHandle);
    isCBV = true;

    // Calculate the aligned buffer size (256-byte alignment required for CBV)
    UINT bufferSize = (numElements * elementSize + 255) & ~255; // Round up to 256 bytes

    // Step 1: Create a committed resource for the CBV (Upload heap for CPU access)
    D3D12_HEAP_PROPERTIES heapProps = {};
    heapProps.Type = D3D12_HEAP_TYPE_UPLOAD;

    D3D12_RESOURCE_DESC bufferDesc = {};
    bufferDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    bufferDesc.Width = bufferSize;
    bufferDesc.Height = 1;
    bufferDesc.DepthOrArraySize = 1;
    bufferDesc.MipLevels = 1;
    bufferDesc.SampleDesc.Count = 1;
    bufferDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    bufferDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

    HRESULT hr = context.getDevice()->CreateCommittedResource(
        &heapProps,
        D3D12_HEAP_FLAG_NONE,
        &bufferDesc,
        D3D12_RESOURCE_STATE_GENERIC_READ,  // State for CPU-read access
        nullptr,
        IID_PPV_ARGS(&buffer)
    );
    if (FAILED(hr)) {
        throw std::runtime_error("Failed to create CBV buffer.");
    }

    // Step 2: Map the buffer and copy data directly into it (no upload buffer needed for CBV)
    void* mappedData = nullptr;
    buffer->Map(0, nullptr, &mappedData);
    memcpy(mappedData, data, numElements * elementSize); // Copy only actual data size
    buffer->Unmap(0, nullptr);

    // Step 3: Create the CBV in the descriptor heap
    D3D12_CONSTANT_BUFFER_VIEW_DESC cbvDesc = {};
    cbvDesc.BufferLocation = buffer->GetGPUVirtualAddress();
    cbvDesc.SizeInBytes = bufferSize; // Must be 256-byte aligned

    context.getDevice()->CreateConstantBufferView(&cbvDesc, CBVcpuHandle);
}

void StructuredBuffer::passDataToGPU(DXContext& context, ID3D12GraphicsCommandList6* cmdList, CommandListID cmdId) {
	// THIS FUNCTION WILL RESET THE COMMAND LIST AT THE END OF THE CALL

	if (isCBV) {
		throw std::runtime_error("Cannot create UAV or SRV after creating CBV.");
	}
    else if (isUAV || isSRV) {
		throw std::runtime_error("UAV or SRV already created.");
	}

    // Calculate the total buffer size
    UINT bufferSize = numElements * elementSize;

    // Step 1: Create a default heap resource for the SRV (GPU-only)
    D3D12_HEAP_PROPERTIES defaultHeapProps = {};
    defaultHeapProps.Type = D3D12_HEAP_TYPE_DEFAULT;

    D3D12_RESOURCE_DESC bufferDesc = {};
    bufferDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    bufferDesc.Width = bufferSize;
    bufferDesc.Height = 1;
    bufferDesc.DepthOrArraySize = 1;
    bufferDesc.MipLevels = 1;
    bufferDesc.SampleDesc.Count = 1;
    bufferDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    bufferDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;  // No unordered access flag

    HRESULT hr = context.getDevice()->CreateCommittedResource(
        &defaultHeapProps,
        D3D12_HEAP_FLAG_NONE,
        &bufferDesc,
        D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr,
        IID_PPV_ARGS(&buffer)
    );
    if (FAILED(hr)) {
        throw std::runtime_error("Failed to create SRV buffer.");
    }

    // Step 2: Create an upload buffer to copy `data` to the GPU buffer
    D3D12_HEAP_PROPERTIES uploadHeapProps = {};
    uploadHeapProps.Type = D3D12_HEAP_TYPE_UPLOAD;

    D3D12_RESOURCE_DESC uploadDesc = bufferDesc;
    uploadDesc.Flags = D3D12_RESOURCE_FLAG_NONE; // Upload buffer has no flags

    ComPointer<ID3D12Resource> uploadBuffer;
    hr = context.getDevice()->CreateCommittedResource(
        &uploadHeapProps,
        D3D12_HEAP_FLAG_NONE,
        &uploadDesc,
        D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr,
        IID_PPV_ARGS(&uploadBuffer)
    );
    if (FAILED(hr)) {
        throw std::runtime_error("Failed to create SRV upload buffer.");
    }

    // Step 3: Copy data into the upload buffer
    void* mappedData = nullptr;
    uploadBuffer->Map(0, nullptr, &mappedData);
    memcpy(mappedData, data, bufferSize); // Assume `data` is correctly sized
    uploadBuffer->Unmap(0, nullptr);

    // Step 4: Copy data from the upload buffer to the GPU buffer
    cmdList->CopyResource(buffer.Get(), uploadBuffer.Get());

    // Step 5: Transition the buffer to the SHADER_RESOURCE state
    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = buffer.Get();
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    cmdList->ResourceBarrier(1, &barrier);

    // Create a fence to wait for the GPU to finish copying data
    ComPointer<ID3D12Fence> fence;
    UINT64 fenceValue = 1;
    hr = context.getDevice()->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
    if (FAILED(hr)) {
        throw std::runtime_error("Failed to create fence.");
    }

    context.executeCommandList(cmdId);
    context.getCommandQueue()->Signal(fence.Get(), fenceValue);

    // Wait for the fence to reach the signaled value
    if (fence->GetCompletedValue() < fenceValue) {
        HANDLE eventHandle = CreateEvent(nullptr, FALSE, FALSE, nullptr);
        if (eventHandle == nullptr) {
            throw std::runtime_error("Failed to create event handle.");
        }

        // Set the event to be triggered when the GPU reaches the fence value
        fence->SetEventOnCompletion(fenceValue, eventHandle);

        // Wait until the event is triggered, meaning the GPU has finished
        WaitForSingleObject(eventHandle, INFINITE);
        CloseHandle(eventHandle);
    }

    context.resetCommandList(cmdId);
}

void StructuredBuffer::createUAV(DXContext& context, DescriptorHeap* dh) {

	if (isCBV) {
		throw std::runtime_error("Cannot create UAV after creating CBV.");
	}
    else if (isUAV) {
		throw std::runtime_error("UAV already created.");
	}

	findFreeHandle(dh, UAVcpuHandle, UAVgpuHandle);

    // Create the UAV in the descriptor heap
    D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
    uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    uavDesc.Buffer.FirstElement = 0;
    uavDesc.Buffer.NumElements = numElements;
    uavDesc.Buffer.StructureByteStride = elementSize;

    context.getDevice()->CreateUnorderedAccessView(buffer.Get(), nullptr, &uavDesc, UAVcpuHandle);

	isUAV = true;
}

void StructuredBuffer::createSRV(DXContext& context, DescriptorHeap* dh) {

	if (isCBV) {
		throw std::runtime_error("Cannot create SRV after creating CBV.");
	}
	else if (isSRV) {
		throw std::runtime_error("SRV already created.");
	}

	findFreeHandle(dh, SRVcpuHandle, SRVgpuHandle);

	// Create the SRV in the descriptor heap
	D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
	srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
	srvDesc.Format = DXGI_FORMAT_UNKNOWN;
	srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
	srvDesc.Buffer.FirstElement = 0;
	srvDesc.Buffer.NumElements = numElements;
	srvDesc.Buffer.StructureByteStride = elementSize;

	context.getDevice()->CreateShaderResourceView(buffer.Get(), &srvDesc, SRVcpuHandle);

	isSRV = true;
}

void StructuredBuffer::copyDataFromGPU(DXContext& context, void* outputData, ID3D12GraphicsCommandList6* cmdList, D3D12_RESOURCE_STATES state, CommandListID cmdId) {
	// THIS FUNCTION WILL RESET THE COMMAND LIST AT THE END OF THE CALL

    // Create a readback buffer to copy data from the GPU buffer
    ComPointer<ID3D12Resource> readbackBuffer;

    // Set up heap properties for a readback buffer
    D3D12_HEAP_PROPERTIES heapProps = {};
    heapProps.Type = D3D12_HEAP_TYPE_READBACK;  // CPU-accessible for readback

    // Describe the buffer resource
    D3D12_RESOURCE_DESC bufferDesc = {};
    bufferDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    bufferDesc.Width = elementSize * numElements;  // Match the size of the GPU buffer
    bufferDesc.Height = 1;
    bufferDesc.DepthOrArraySize = 1;
    bufferDesc.MipLevels = 1;
    bufferDesc.SampleDesc.Count = 1;
    bufferDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    bufferDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

    // Create the readback buffer resource
    HRESULT hr = context.getDevice()->CreateCommittedResource(
        &heapProps,
        D3D12_HEAP_FLAG_NONE,
        &bufferDesc,
        D3D12_RESOURCE_STATE_COPY_DEST,  // Must be in COPY_DEST state to receive copied data
        nullptr,
        IID_PPV_ARGS(&readbackBuffer)
    );

    if (FAILED(hr)) {
        throw std::runtime_error("Failed to create readback buffer.");
    }

    // 2. Transition the buffer to COPY_SOURCE state
    D3D12_RESOURCE_BARRIER uavToCopySourceBarrier = {};
    uavToCopySourceBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    uavToCopySourceBarrier.Transition.pResource = buffer.Get();
    uavToCopySourceBarrier.Transition.StateBefore = state;
    uavToCopySourceBarrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    uavToCopySourceBarrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    cmdList->ResourceBarrier(1, &uavToCopySourceBarrier);

    // Copy the data from the GPU buffer to the readback buffer
    cmdList->CopyResource(readbackBuffer.Get(), buffer.Get());

	// Transition the buffer back to its original state
    D3D12_RESOURCE_BARRIER copySourceToUavBarrier = {};
    copySourceToUavBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    copySourceToUavBarrier.Transition.pResource = buffer.Get();
    copySourceToUavBarrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
    copySourceToUavBarrier.Transition.StateAfter = state;
    copySourceToUavBarrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    cmdList->ResourceBarrier(1, &copySourceToUavBarrier);

    // Create a fence
    UINT64 fenceValue = 1;
    ComPointer<ID3D12Fence> fence;
    context.getDevice()->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));

    // Execute the command list to perform the copy operation
    context.executeCommandList(cmdId);
    context.getCommandQueue()->Signal(fence.Get(), fenceValue);

    if (fence->GetCompletedValue() < fenceValue) {
        HANDLE eventHandle = CreateEvent(nullptr, FALSE, FALSE, nullptr);
        if (eventHandle == nullptr) {
            throw std::runtime_error("Failed to create event handle.");
        }

        // Set the event to be triggered when the GPU reaches the fence value
        fence->SetEventOnCompletion(fenceValue, eventHandle);

        // Wait until the event is triggered, meaning the GPU has finished
        WaitForSingleObject(eventHandle, INFINITE);
        CloseHandle(eventHandle);
    }

    context.resetCommandList(cmdId);

    // Map the readback buffer to access the data on the CPU
    void* mappedData = nullptr;
    D3D12_RANGE readRange{ 0, elementSize * numElements }; // The range of the buffer to map
    hr = readbackBuffer->Map(0, &readRange, &mappedData);
    if (FAILED(hr)) {
        throw std::runtime_error("Failed to map readback buffer.");
    }
    // Copy data from the mapped readback buffer to outputData
    memcpy(outputData, mappedData, elementSize * numElements);

    // Unmap the readback buffer
    D3D12_RANGE writeRange{ 0, 0 }; // Indicate no data written by CPU
    readbackBuffer->Unmap(0, &writeRange);

}

ComPointer<ID3D12Resource1>& StructuredBuffer::getBuffer()
{
	return this->buffer;
}

void StructuredBuffer::releaseResources()
{
	this->buffer.Release();
}