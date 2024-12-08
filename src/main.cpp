#include "main.h"

int main() {
    //set up DX, window, keyboard mouse
    DebugLayer debugLayer = DebugLayer();
    DXContext context = DXContext();
    std::unique_ptr<Camera> camera = std::make_unique<Camera>();
    std::unique_ptr<Keyboard> keyboard = std::make_unique<Keyboard>();
    std::unique_ptr<Mouse> mouse = std::make_unique<Mouse>();

    if (!Window::get().init(&context, SCREEN_WIDTH, SCREEN_HEIGHT)) {
        //handle could not initialize window
        std::cout << "could not initialize window\n";
        Window::get().shutdown();
        return false;
    }

    //initialize ImGUI
    ImGuiIO& io = initImGUI(context);

    //set mouse to use the window
    mouse->SetWindow(Window::get().getHWND());

    //initialize scene
    Scene scene{camera.get(), &context};

    PBMPMConstants pbmpmCurrConstants = scene.getPBMPMConstants();
    PBMPMConstants pbmpmIterConstants = pbmpmCurrConstants;

    unsigned int renderMeshlets = 0;
    unsigned int renderMode = 0;

    while (!Window::get().getShouldClose()) {
        //update window
        Window::get().update();
        if (Window::get().getShouldResize()) {
            //flush pending buffer operations in swapchain
            context.flush(FRAME_COUNT);
            Window::get().resize();
            camera->updateAspect((float)Window::get().getWidth() / (float)Window::get().getHeight());
        }

        auto kState = keyboard->GetState();
        auto mState = mouse->GetState();
        mouse->SetMode(mState.leftButton ? Mouse::MODE_RELATIVE : Mouse::MODE_ABSOLUTE);
        camera->kmStateCheck(kState, mState);

        if (mState.rightButton) {
            //enable mouse force
            pbmpmIterConstants.mouseActivation = 1;

            POINT cursorPos;
            GetCursorPos(&cursorPos);

            float ndcX = (2.0f * cursorPos.x) / SCREEN_WIDTH - 1.0f;
            float ndcY = -(2.0f * cursorPos.y) / SCREEN_HEIGHT + 1.0f;

            XMVECTOR screenCursorPos = XMVectorSet(ndcX, ndcY, 0.0f, 1.0f);
            XMVECTOR worldCursorPos = XMVector4Transform(screenCursorPos, camera->getInvViewProjMat());
            XMStoreFloat4(&(pbmpmIterConstants.mousePosition), worldCursorPos);

            pbmpmIterConstants.mouseFunction = 0;
            pbmpmIterConstants.mouseRadius = 1000;
            scene.updatePBMPMConstants(pbmpmIterConstants);
        }
        else {
            pbmpmIterConstants.mouseActivation = 0;
        }

        //compute pbmpm + mesh shader
        scene.compute();

        //get pipelines
        auto renderPipeline = scene.getPBMPMRenderPipeline();
        auto meshPipeline = scene.getFluidMeshPipeline();
        auto objectPipeline = scene.getObjectPipeline();
        //whichever pipeline renders first should begin and end the frame
        auto firstPipeline = meshPipeline;

        //begin frame
        Window::get().beginFrame(firstPipeline->getCommandList());

        //create viewport
        D3D12_VIEWPORT vp;
        Window::get().createViewport(vp, firstPipeline->getCommandList());

        //mesh render pass
        Window::get().setRT(meshPipeline->getCommandList());
        Window::get().setViewport(vp, meshPipeline->getCommandList());
        if (renderMode != 1) scene.drawFluid(renderMeshlets);
        context.executeCommandList(meshPipeline->getCommandListID());

        //first render pass
        Window::get().setRT(renderPipeline->getCommandList());
        Window::get().setViewport(vp, renderPipeline->getCommandList());
        if (renderMode != 0) scene.drawPBMPM();

        //set up ImGUI for frame
        ImGui_ImplDX12_NewFrame();
        ImGui_ImplWin32_NewFrame();
        ImGui::NewFrame();

        //draw ImGUI
        drawImGUIWindow(pbmpmIterConstants, io, &renderMeshlets, &renderMode, scene.getFluidIsovalue(), scene.getFluidKernelScale());

        //render ImGUI
        ImGui::Render();
        if (!PBMPMScene::constantsEqual(pbmpmIterConstants, pbmpmCurrConstants)) {
            scene.updatePBMPMConstants(pbmpmIterConstants);
            pbmpmCurrConstants = pbmpmIterConstants;
        }

        renderPipeline->getCommandList()->SetDescriptorHeaps(1, &imguiSRVHeap);
        ImGui_ImplDX12_RenderDrawData(ImGui::GetDrawData(), renderPipeline->getCommandList());

        context.executeCommandList(renderPipeline->getCommandListID());

        //object render pass
        Window::get().setRT(objectPipeline->getCommandList());
        Window::get().setViewport(vp, objectPipeline->getCommandList());
        if (renderGrid) scene.drawObject();
        context.executeCommandList(objectPipeline->getCommandListID());

        //end frame
        Window::get().endFrame(firstPipeline->getCommandList());

        Window::get().present();
		context.resetCommandList(renderPipeline->getCommandListID());
        context.resetCommandList(meshPipeline->getCommandListID());
        context.resetCommandList(objectPipeline->getCommandListID());
    }

    // Scene should release all resources, including their pipelines
    scene.releaseResources();

    ImGui_ImplDX12_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImGui::DestroyContext();

    imguiSRVHeap->Release();

    //flush pending buffer operations in swapchain
    context.flush(FRAME_COUNT);
    Window::get().shutdown();
}
