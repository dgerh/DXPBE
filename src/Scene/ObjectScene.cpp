#include "ObjectScene.h"
#include "GridConstants.h"

ObjectScene::ObjectScene(DXContext* context, RenderPipeline* pipeline, std::vector<SimShape>& shapes)
	: Drawable(context, pipeline), shapes(shapes)
{
	constructScene();
}

void ObjectScene::constructScene()
{
	renderPipeline->createPSOD();
	renderPipeline->createPipelineState(context->getDevice());

	inputStrings.push_back("objs\\cube.obj");

    XMFLOAT4X4 gridModelMatrix;
    XMStoreFloat4x4(&gridModelMatrix, XMMatrixScaling(GRID_WIDTH, GRID_HEIGHT, GRID_DEPTH));
    modelMatrices.push_back(gridModelMatrix);

    
    for (const auto& shape : shapes) {
        inputStrings.push_back("objs\\cube.obj");
        XMFLOAT4X4 simShapeMatrix;
        XMStoreFloat4x4(&simShapeMatrix, XMMatrixMultiply(

            XMMatrixScaling(shape.halfSize.x * 2, shape.halfSize.y * 2, shape.halfSize.z * 2),
            XMMatrixTranslation(shape.position.x - shape.halfSize.x, shape.position.y - shape.halfSize.y, shape.position.z - shape.halfSize.z)
        ));
        modelMatrices.push_back(simShapeMatrix);
    }

    for (int i = 0; i < inputStrings.size(); i++) {
        auto string = inputStrings.at(i);
        auto m = modelMatrices.at(i);
		Mesh newMesh = Mesh((std::filesystem::current_path() / string).string(), context, renderPipeline->getCommandList(), renderPipeline, m, true);
		meshes.push_back(newMesh);
		sceneSize += newMesh.getNumTriangles();
	}
}

void ObjectScene::draw(Camera* camera) {
    for (Mesh m : meshes) {
        // == IA ==
        auto cmdList = renderPipeline->getCommandList();
        cmdList->IASetVertexBuffers(0, 1, m.getVBV());
        cmdList->IASetIndexBuffer(m.getIBV());
        //cmdList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        cmdList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_LINELIST);
        // == RS ==
        //NO NEED TO RESET VIEWPORT??
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
        //model mat for this mesh
        cmdList->SetGraphicsRoot32BitConstants(0, 16, m.getModelMatrix(), 32);

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