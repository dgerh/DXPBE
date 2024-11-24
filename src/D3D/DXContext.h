#pragma once
#include "../Support/WinInclude.h"
#include "../Support/ComPointer.h"
#include <stdexcept>
#include <array>

#define NUM_CMDLISTS 8
enum CommandListID {
    PAPA_ID,
    RENDER_ID,
    PBMPM_COMPUTE_ID,
    MESH_ID,
    PBD_ID,
    apply_force_ID,
    velocity_update_ID,
    FaceToFace_ID,

};

class DXContext
{
public:
    DXContext();
    ~DXContext();

    void signalAndWait();
    void resetCommandList(CommandListID id);
	void executeCommandList(CommandListID id);

    void flush(size_t count);
    void signalAndWaitForFence(ComPointer<ID3D12Fence>& fence, UINT64& fenceValue);

    ComPointer<IDXGIFactory7>& getFactory();
    ComPointer<ID3D12Device6>& getDevice();
    ComPointer<ID3D12CommandQueue>& getCommandQueue();
    ComPointer<ID3D12CommandAllocator>& getCommandAllocator(CommandListID id) { return cmdAllocators[id]; };
    ID3D12GraphicsCommandList6* createCommandList(CommandListID id);

private:
    ComPointer<IDXGIFactory7> dxgiFactory;

    ComPointer<ID3D12Device6> device;

    ComPointer<ID3D12CommandQueue> cmdQueue;
    std::array<ComPointer<ID3D12CommandAllocator>, NUM_CMDLISTS> cmdAllocators{};
    std::array<ComPointer<ID3D12GraphicsCommandList6>, NUM_CMDLISTS> cmdLists{};

    ComPointer<ID3D12Fence1> fence;
    UINT64 fenceValue = 0;
    HANDLE fenceEvent = nullptr;

};