#include <iostream>

#include "Support/WinInclude.h"
#include "Support/ComPointer.h"
#include "Support/Window.h"
#include "Support/Shader.h"

#include "Debug/DebugLayer.h"

#include "D3D/DXContext.h"
#include "D3D/Pipeline/RenderPipeline.h"
#include "D3D/Pipeline/MeshPipeline.h"
#include "D3D/Pipeline/ComputePipeline.h"


#include "Scene/Camera.h"
#include "Scene/Scene.h"
#include "Scene/PBMPMScene.h"
//#include "Scene/PBD.h"

#include "ImGUI/ImGUIHelper.h"

static ImGUIDescriptorHeapAllocator imguiHeapAllocator;
static ID3D12DescriptorHeap* imguiSRVHeap = nullptr;

static bool meshletRenderType = false;
static unsigned int renderModeType = 0; // 0 = both particles and mesh shading, 1 = just mesh shading, 2 = just particles
static int fixedPointExponent = 7;
static bool useGridVolume = true;
static bool renderGrid = false;

const char* modes[] = { "Mesh Shaded Fluid, Non-Fluid Particles", "Mesh Shaded Fluid, All Particles", "No Mesh Shaded Fluid, All Particles"};

ImGuiIO& initImGUI(DXContext& context) {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

    ImGui::StyleColorsDark();

    ImGui_ImplWin32_Init(Window::get().getHWND());

    ImGui_ImplDX12_InitInfo imguiDXInfo;
    imguiDXInfo.CommandQueue = context.getCommandQueue();
    imguiDXInfo.Device = context.getDevice();
    imguiDXInfo.NumFramesInFlight = 2;
    imguiDXInfo.RTVFormat = DXGI_FORMAT_R8G8B8A8_UNORM;

    D3D12_DESCRIPTOR_HEAP_DESC desc = {};
    desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    desc.NumDescriptors = 64;
    desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    if (context.getDevice()->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&imguiSRVHeap)) != S_OK) {
        std::cout << "could not create imgui descriptor heap\n";
        Window::get().shutdown();
    }
    imguiHeapAllocator.Create(context.getDevice(), imguiSRVHeap);

    imguiHeapAllocator.Heap = imguiSRVHeap;
    imguiDXInfo.SrvDescriptorHeap = imguiSRVHeap;
    imguiDXInfo.SrvDescriptorAllocFn = [](ImGui_ImplDX12_InitInfo*, D3D12_CPU_DESCRIPTOR_HANDLE* out_cpu_handle, D3D12_GPU_DESCRIPTOR_HANDLE* out_gpu_handle) { return imguiHeapAllocator.Alloc(out_cpu_handle, out_gpu_handle); };
    imguiDXInfo.SrvDescriptorFreeFn = [](ImGui_ImplDX12_InitInfo*, D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle, D3D12_GPU_DESCRIPTOR_HANDLE gpu_handle) { return imguiHeapAllocator.Free(cpu_handle, gpu_handle); };

    ImGui_ImplDX12_Init(&imguiDXInfo);

    return io;
}

void drawImGUIWindow(PBMPMConstants& pbmpmConstants, ImGuiIO& io, unsigned int* renderMeshlets, unsigned int* renderMode, float* isovalue, float* kernelScale, float* kernelRadius, unsigned int* substepCount, int numParticles) {
    ImGui::Begin("Scene Options");

    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);
    ImGui::Text("Number of Particles: %d", numParticles);

    if (ImGui::CollapsingHeader("Simulation Parameters")) {
        ImGui::SliderFloat("Gravity Strength", &pbmpmConstants.gravityStrength, 0.0f, 20.0f);
        ImGui::SliderFloat("Liquid Relaxation", &pbmpmConstants.liquidRelaxation, 0.0f, 10.0f);
        ImGui::SliderFloat("Liquid Viscosity", &pbmpmConstants.liquidViscosity, 0.0f, 1.0f);
        ImGui::SliderFloat("Friction Angle", &pbmpmConstants.frictionAngle, 0.0f, 90.0f);

        ImGui::SliderFloat("Elastic Relaxation", &pbmpmConstants.elasticRelaxation, 0.0f, 10.0f);
        ImGui::SliderFloat("Elastic Ratio", &pbmpmConstants.elasticityRatio, 0.0f, 2.0f);

        ImGui::SliderInt("Particles Per Cell Axis", (int*)&pbmpmConstants.particlesPerCellAxis, 1, 8);
        ImGui::SliderInt("Fixed Point Multiplier", (int*)&fixedPointExponent, 4, 13);
        pbmpmConstants.fixedPointMultiplier = (unsigned int)pow(10, fixedPointExponent);

        ImGui::SliderFloat("Border Friction", &pbmpmConstants.borderFriction, 0.0f, 1.0f);

        ImGui::SliderInt("Iteration Count", (int*)&pbmpmConstants.iterationCount, 1, 10);
        ImGui::SliderInt("Substep Count", (int*)substepCount, 1, 20);

        ImGui::SliderInt("Mouse Radius", (int*)&pbmpmConstants.mouseRadius, 1, 10);
        ImGui::SliderFloat("Mouse Velocity", &pbmpmConstants.mouseVelocity, 0.f, 300.f);

        ImGui::Checkbox("Use Grid Volume for Liquid", (bool*)&useGridVolume);
        pbmpmConstants.useGridVolumeForLiquid = useGridVolume;
    }

    if (ImGui::CollapsingHeader("Mesh Shading Parameters")) {
        ImGui::TextColored({ 0, 1, 0, 1 }, "Mesh Shading Parameters");
        ImGui::SliderFloat("Isovalue", isovalue, 0.01f, 1.0f);
        ImGui::SliderFloat("Kernel Scale", kernelScale, 2.5f, 12.0f);
        ImGui::SliderFloat("Kernel Radius", kernelRadius, 0.3f, 3.0f);
    }

    if (ImGui::CollapsingHeader("Render Parameters")) {
        if (ImGui::Combo("Select Render Mode", (int*)&renderModeType, modes, IM_ARRAYSIZE(modes)))
        {
            *renderMode = renderModeType;
        }

        ImGui::Checkbox("Render Grid", &renderGrid);

        ImGui::Checkbox("Render Meshlets", &meshletRenderType);
        *renderMeshlets = meshletRenderType;
    }

    ImGui::End();
}

void ComputeMouseRay(
    HWND hwnd,
    float ndcX,
    float ndcY,
    const XMMATRIX& projectionMatrix,
    const XMMATRIX& viewMatrix,
    XMFLOAT4& rayOrigin,
    XMFLOAT4& rayDirection
) {
    // Invert the projection and view matrices
    XMMATRIX invProj = XMMatrixInverse(nullptr, projectionMatrix);
    XMMATRIX invView = XMMatrixInverse(nullptr, viewMatrix);

    // Define the mouse's NDC position on the near and far planes
    XMVECTOR ndcNear = XMVectorSet(ndcX, ndcY, 0.0f, 1.0f); // Near plane
    XMVECTOR ndcFar = XMVectorSet(ndcX, ndcY, 1.0f, 1.0f);   // Far plane

    // Unproject the NDC points to view space
    XMVECTOR viewNear = XMVector3TransformCoord(ndcNear, invProj);
    XMVECTOR viewFar = XMVector3TransformCoord(ndcFar, invProj);

    // Transform the points from view space to world space
    XMVECTOR worldNear = XMVector3TransformCoord(viewNear, invView);
    XMVECTOR worldFar = XMVector3TransformCoord(viewFar, invView);

    // Calculate the ray origin (camera position) and direction
    rayOrigin = XMFLOAT4(
        XMVectorGetX(worldNear),
        XMVectorGetY(worldNear),
        XMVectorGetZ(worldNear),
        1.0f
    );

    XMVECTOR rayDir = XMVector3Normalize(XMVectorSubtract(worldFar, worldNear));
    rayDirection = XMFLOAT4(
        XMVectorGetX(rayDir),
        XMVectorGetY(rayDir),
        XMVectorGetZ(rayDir),
        0.0f
    );
}
