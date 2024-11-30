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
    Scene scene{PBMPM, camera.get(), &context};

    PBMPMConstants pbmpmConstants{ {512, 512, 512}, 0.01F, 2.5F, 1.5F, 0.01F,
        (unsigned int)std::ceil(std::pow(10, 7)),
        1, 4, 30, 0, 0, 0, 0, 0, 0, 5, 0.9F };
    PBMPMConstants pbmpmTempConstants = pbmpmConstants;

    while (!Window::get().getShouldClose()) {
        //update window
        Window::get().update();
        if (Window::get().getShouldResize()) {
            //flush pending buffer operations in swapchain
            context.flush(FRAME_COUNT);
            Window::get().resize();
            camera->updateAspect((float)Window::get().getWidth() / (float)Window::get().getHeight());
        }

        //check keyboard state
        auto kState = keyboard->GetState();
        if (kState.W) {
            camera->translate({ 0.f, 0.f, 1.0f });
        }
        if (kState.A) {
            camera->translate({ -1.0f, 0.f, 0.f });
        }
        if (kState.S) {
            camera->translate({ 0.f, 0.f, -1.0f });
        }
        if (kState.D) {
            camera->translate({ 1.0f, 0.f, 0.f });
        }
        if (kState.Space) {
            camera->translate({ 0.f, 1.0f, 0.f });
        }
        if (kState.LeftControl) {
            camera->translate({ 0.f, -1.0f, 0.f });
        }
        if (kState.D1) {
            scene.setRenderScene(Object);
        }
        if (kState.D2) {
            scene.setRenderScene(PBMPM);
        }
        if (kState.D3) {
            scene.setRenderScene(Fluid);
        }

        //check mouse state
        auto mState = mouse->GetState();

        mouse->SetMode(mState.leftButton ? Mouse::MODE_RELATIVE : Mouse::MODE_ABSOLUTE);

        if (mState.positionMode == Mouse::MODE_RELATIVE && kState.LeftShift) {
            camera->rotateOnX(-mState.y * 0.01f);
            camera->rotateOnY(mState.x * 0.01f);
            camera->rotate();
        }

        //update camera
        camera->updateViewMat();

        //draw to window
        auto renderPipeline = scene.getRenderPipeline();
        scene.compute();

        //begin frame
        Window::get().beginFrame(renderPipeline->getCommandList());
        D3D12_VIEWPORT vp;
        Window::get().createAndSetDefaultViewport(vp, renderPipeline->getCommandList());

        //draw scene
        scene.draw();

        //set up ImGUI for frame
        ImGui_ImplDX12_NewFrame();
        ImGui_ImplWin32_NewFrame();
        ImGui::NewFrame();

        //draw ImGUI
        drawImGUIWindow(pbmpmTempConstants, io);

        //render ImGUI
        ImGui::Render();
        if (!PBMPMScene::constantsEqual(pbmpmTempConstants, pbmpmConstants)) {
            scene.updatePBMPMConstants(pbmpmTempConstants);
            pbmpmConstants = pbmpmTempConstants;
        }

        renderPipeline->getCommandList()->SetDescriptorHeaps(1, &imguiSRVHeap);
        ImGui_ImplDX12_RenderDrawData(ImGui::GetDrawData(), renderPipeline->getCommandList());

        Window::get().endFrame(renderPipeline->getCommandList());

        //finish draw, present, reset
        context.executeCommandList(renderPipeline->getCommandListID());

        Window::get().present();
		context.resetCommandList(renderPipeline->getCommandListID());
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
