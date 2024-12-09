#include "ObjectScene.h"
#include "SceneConstants.h"

ObjectScene::ObjectScene(DXContext* context, RenderPipeline* pipeline, std::vector<SimShape>& shapes, bool isWireframeScene)
	: Drawable(context, pipeline), shapes(shapes)
{
    if (isWireframeScene) {
        constructSceneWire();
    }
    else {
        constructSceneSolid();
    }
}

void ObjectScene::constructSceneWire() {
	renderPipeline->createPSOD();
	renderPipeline->createPipelineState(context->getDevice());

	inputStrings.push_back("objs\\cube.obj");

    //cube for grid
    XMFLOAT4X4 gridModelMatrix;
    XMStoreFloat4x4(&gridModelMatrix, XMMatrixScaling(GRID_WIDTH, GRID_HEIGHT, GRID_DEPTH));
    modelMatrices.push_back(gridModelMatrix);
    
    //cubes for shapes
    for (const auto& shape : shapes) {
        inputStrings.push_back("objs\\cube.obj");
        XMFLOAT4X4 simShapeMatrix;
        XMStoreFloat4x4(&simShapeMatrix, XMMatrixMultiply(
            XMMatrixScaling(shape.halfSize.x * 2, shape.halfSize.y * 2, shape.halfSize.z * 2),
            XMMatrixTranslation(shape.position.x - shape.halfSize.x, shape.position.y - shape.halfSize.y, shape.position.z - shape.halfSize.z)
        ));
        modelMatrices.push_back(simShapeMatrix);
    }

    auto string = inputStrings.front();
    auto m = modelMatrices.front();
    Mesh newMesh = Mesh((std::filesystem::current_path() / string).string(), context, renderPipeline->getCommandList(), renderPipeline, m, true, {0, 1, 0});
    meshes.push_back(newMesh);
    sceneSize += newMesh.getNumTriangles();

    //push shapes as wireframe
    for (int i = 1; i < inputStrings.size(); i++) {
        auto string = inputStrings.at(i);
        auto m = modelMatrices.at(i);
		Mesh newMesh = Mesh((std::filesystem::current_path() / string).string(), context, renderPipeline->getCommandList(), renderPipeline, m, true, { 1, 0, 0 });
		meshes.push_back(newMesh);
		sceneSize += newMesh.getNumTriangles();
	}
}

void ObjectScene::constructSceneSolid() {
    //cube for ground
    inputStrings.push_back("objs\\cube.obj");
    XMFLOAT4X4 groundModelMatrix;
    XMStoreFloat4x4(&groundModelMatrix, XMMatrixScaling(1.5 * GRID_WIDTH, -5.0f, 1.5 * GRID_DEPTH));
    modelMatrices.push_back(groundModelMatrix);

    //push ground as solid
    auto string = inputStrings.back();
    auto m = modelMatrices.back();
    Mesh newMesh = Mesh((std::filesystem::current_path() / string).string(), context, renderPipeline->getCommandList(), renderPipeline, m, false, XMFLOAT3(GROUND_PLANE_COLOR));
    meshes.push_back(newMesh);
    sceneSize += newMesh.getNumTriangles();
}

void ObjectScene::draw(Camera* camera) {
    for (Mesh m : meshes) {
        // == IA ==
        auto cmdList = renderPipeline->getCommandList();
        cmdList->IASetVertexBuffers(0, 1, m.getVBV());
        cmdList->IASetIndexBuffer(m.getIBV());

        if (m.getIsWireframe()) {
            cmdList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_LINELIST);
        }
        else {
            cmdList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        }

        // == PSO ==
        cmdList->SetPipelineState(renderPipeline->getPSO());
        cmdList->SetGraphicsRootSignature(renderPipeline->getRootSignature());
        
        // == ROOT ==
        ID3D12DescriptorHeap* descriptorHeaps[] = { renderPipeline->getDescriptorHeap()->GetAddress() };
        cmdList->SetDescriptorHeaps(_countof(descriptorHeaps), descriptorHeaps);
        cmdList->SetGraphicsRootDescriptorTable(1, renderPipeline->getDescriptorHeap()->GetGPUHandleAt(0)); // Descriptor table slot 1 for CBV

        auto viewMat = camera->getViewMat();
        auto projMat = camera->getProjMat();
        cmdList->SetGraphicsRoot32BitConstants(0, 16, &viewMat, 0);
        cmdList->SetGraphicsRoot32BitConstants(0, 16, &projMat, 16);
        cmdList->SetGraphicsRoot32BitConstants(0, 16, m.getModelMatrix(), 32);
        cmdList->SetGraphicsRoot32BitConstants(0, 3, m.getColor(), 48);

        cmdList->DrawIndexedInstanced(m.getNumTriangles() * 3, 1, 0, 0, 0);
    }
}

size_t ObjectScene::getSceneSize() {
    return sceneSize;
}

void ObjectScene::releaseResources() {
	for (Mesh m : meshes) {
		m.releaseResources();
	}
    renderPipeline->releaseResources();
}