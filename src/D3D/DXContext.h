#pragma once
#include "../Support/WinInclude.h"
#include "../Support/ComPointer.h"
#include <stdexcept>
#include <array>

#define NUM_CMDLISTS 26
enum CommandListID {
    OBJECT_RENDER_WIRE_ID,
    OBJECT_RENDER_SOLID_ID,
    PBMPM_RENDER_ID,
    PBMPM_G2P2G_COMPUTE_ID,
    PBMPM_BUKKITCOUNT_COMPUTE_ID,
    PBMPM_BUKKITALLOCATE_COMPUTE_ID,
    PBMPM_BUKKITINSERT_COMPUTE_ID,
    PBMPM_BUFFERCLEAR_COMPUTE_ID,
    PBMPM_EMISSION_COMPUTE_ID,
    PBMPM_SET_INDIRECT_ARGS_COMPUTE_ID,
    PHYSICS_RENDER_ID,
    PHYSICS_COMPUTE_ID,
    FLUID_RENDER_ID,
    BILEVEL_UNIFORM_GRID_COMPUTE_ID,
    SURFACE_BLOCK_DETECTION_COMPUTE_ID,
    SURFACE_CELL_DETECTION_COMPUTE_ID,
    SURFACE_VERTEX_COMPACTION_COMPUTE_ID,
    SURFACE_VERTEX_DENSITY_COMPUTE_ID,
    SURFACE_VERTEX_NORMAL_COMPUTE_ID,
    FLUID_BUFFER_CLEAR_COMPUTE_ID,
    FLUID_MESH_ID,

    PBD_Render_ID,
    PBD_ID,
    Gram_ID,
    apply_force_ID,
    velocity_update_ID,
    FaceToFace_ID

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
    double readTimingQueryData();
    void startTimingQuery(ID3D12GraphicsCommandList6* cmdList);
    void endTimingQuery(ID3D12GraphicsCommandList6* cmdList);

private:
    void initTimingResources();

    ComPointer<ID3D12QueryHeap> queryHeap;
    ComPointer<ID3D12Resource> queryResultBuffer;

    ComPointer<IDXGIFactory7> dxgiFactory;

    ComPointer<ID3D12Device6> device;

    ComPointer<ID3D12CommandQueue> cmdQueue;
    std::array<ComPointer<ID3D12CommandAllocator>, NUM_CMDLISTS> cmdAllocators{};
    std::array<ComPointer<ID3D12GraphicsCommandList6>, NUM_CMDLISTS> cmdLists{};

    ComPointer<ID3D12Fence1> fence;
    UINT64 fenceValue = 0;
    HANDLE fenceEvent = nullptr;

};