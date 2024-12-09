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

	// Get the client area of the window
    RECT rect;
    GetClientRect(Window::get().getHWND(), &rect);
    float clientWidth = static_cast<float>(rect.right - rect.left);
    float clientHeight = static_cast<float>(rect.bottom - rect.top);

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

            if (kState.LeftShift) {
                // Pulling Fluid
                pbmpmIterConstants.mouseFunction = 2;
			}
            else if (kState.LeftAlt) {
                // Grab Fluid Ball
                pbmpmIterConstants.mouseFunction = 1;
            }
            else {
				// Pushing Fluid
                pbmpmIterConstants.mouseFunction = 0;
			}

            //enable mouse force
            pbmpmIterConstants.mouseActivation = 1;

            POINT mousePos;
            GetCursorPos(&mousePos);

			ScreenToClient(Window::get().getHWND(), &mousePos);

            float ndcX = (2.0f * mousePos.x / clientWidth) - 1.0f;
            float ndcY = 1.0f - (2.0f * mousePos.y / clientHeight);

			XMFLOAT4 prevMousePos = pbmpmIterConstants.mousePosition;
			ComputeMouseRay(
				Window::get().getHWND(),
				ndcX,
				ndcY,
				camera->getProjMat(),
				camera->getViewMat(),
                pbmpmIterConstants.mousePosition,
                pbmpmIterConstants.mouseDirection
			);

            pbmpmIterConstants.mouseRadius = 5.0;
            pbmpmIterConstants.mouseVelocity = 100.0;

            scene.updatePBMPMConstants(pbmpmIterConstants);
        }
        else {
            pbmpmIterConstants.mouseActivation = 0;
			scene.updatePBMPMConstants(pbmpmIterConstants);
        }

        //compute pbmpm + mesh shader
        scene.compute();

        //get pipelines
        auto renderPipeline = scene.getPBMPMRenderPipeline();
        auto meshPipeline = scene.getFluidMeshPipeline();
        auto objectWirePipeline = scene.getObjectWirePipeline();
        auto objectSolidPipeline = scene.getObjectSolidPipeline();
        //whichever pipeline renders first should begin and end the frame
        auto firstPipeline = objectWirePipeline;

        //begin frame
        Window::get().beginFrame(firstPipeline->getCommandList());

        //create viewport
        D3D12_VIEWPORT vp;
        Window::get().createViewport(vp, firstPipeline->getCommandList());

        //wire object render pass
        Window::get().setRT(objectWirePipeline->getCommandList());
        Window::get().setViewport(vp, objectWirePipeline->getCommandList());
        if (renderGrid) scene.drawWireObjects();
        context.executeCommandList(objectWirePipeline->getCommandListID());

        //solid object render pass
        Window::get().setRT(objectSolidPipeline->getCommandList());
        Window::get().setViewport(vp, objectSolidPipeline->getCommandList());
        scene.drawSolidObjects();
        context.executeCommandList(objectSolidPipeline->getCommandListID());

        //mesh render pass
        Window::get().setRT(meshPipeline->getCommandList());
        Window::get().setViewport(vp, meshPipeline->getCommandList());
        if (renderMode != 1) scene.drawFluid(renderMeshlets);
        context.executeCommandList(meshPipeline->getCommandListID());

        //particles + imgui render pass
        Window::get().setRT(renderPipeline->getCommandList());
        Window::get().setViewport(vp, renderPipeline->getCommandList());
        if (renderMode != 0) scene.drawPBMPM();

        //set up ImGUI for frame
        ImGui_ImplDX12_NewFrame();
        ImGui_ImplWin32_NewFrame();
        ImGui::NewFrame();

        //draw ImGUI
        drawImGUIWindow(pbmpmIterConstants, io, &renderMeshlets, &renderMode, 
            scene.getFluidIsovalue(), 
            scene.getFluidKernelScale(), 
            scene.getFluidKernelRadius(), 
            scene.getPBMPMSubstepCount(),
            scene.getNumParticles());

        //render ImGUI
        ImGui::Render();
        if (!PBMPMScene::constantsEqual(pbmpmIterConstants, pbmpmCurrConstants)) {
            scene.updatePBMPMConstants(pbmpmIterConstants);
            pbmpmCurrConstants = pbmpmIterConstants;
        }

        renderPipeline->getCommandList()->SetDescriptorHeaps(1, &imguiSRVHeap);
        ImGui_ImplDX12_RenderDrawData(ImGui::GetDrawData(), renderPipeline->getCommandList());

        context.executeCommandList(renderPipeline->getCommandListID());

        //end frame
        Window::get().endFrame(firstPipeline->getCommandList());

        Window::get().present();
		context.resetCommandList(renderPipeline->getCommandListID());
        context.resetCommandList(meshPipeline->getCommandListID());
        context.resetCommandList(objectWirePipeline->getCommandListID());
        context.resetCommandList(objectSolidPipeline->getCommandListID());
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
