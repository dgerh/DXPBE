#include "Scene.h"

Scene::Scene(Camera* p_camera, DXContext* context)
	:  camera(p_camera),
	pbmpmRP("PBMPMVertexShader.cso", "PixelShader.cso", "PBMPMVertexRootSignature.cso", *context, CommandListID::PBMPM_RENDER_ID,
		D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE),
	pbmpmIC(100),
	pbmpmScene(context, &pbmpmRP, pbmpmIC),
	objectRP("VertexShader.cso", "PixelShader.cso", "RootSignature.cso", *context, CommandListID::OBJECT_RENDER_ID,
		D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE),
	objectScene(context, &objectRP, pbmpmScene.getSimShapes()),
	fluidRP("VertexShader.cso", "PixelShader.cso", "RootSignature.cso", *context, CommandListID::FLUID_RENDER_ID,
		D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE),
	bilevelUniformGridCP("BilevelUniformGridRootSig.cso", "BilevelUniformGrid.cso", *context, CommandListID::BILEVEL_UNIFORM_GRID_COMPUTE_ID, 
		D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 45, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE),
	surfaceBlockDetectionCP("SurfaceBlockDetectionRootSig.cso", "SurfaceBlockDetection.cso", *context, CommandListID::SURFACE_BLOCK_DETECTION_COMPUTE_ID,
		D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE),
	surfaceCellDetectionCP("SurfaceCellDetectionRootSig.cso", "SurfaceCellDetection.cso", *context, CommandListID::SURFACE_CELL_DETECTION_COMPUTE_ID,
		D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE),
	surfaceVertexCompactionCP("SurfaceVertexCompactionRootSig.cso", "SurfaceVertexCompaction.cso", *context, CommandListID::SURFACE_VERTEX_COMPACTION_COMPUTE_ID,
		D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE),
	surfaceVertexDensityCP("SurfaceVertexDensityRootSig.cso", "SurfaceVertexDensity.cso", *context, CommandListID::SURFACE_VERTEX_DENSITY_COMPUTE_ID,
		D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE),
	surfaceVertexNormalCP("SurfaceVertexNormalsRootSig.cso", "SurfaceVertexNormals.cso", *context, CommandListID::SURFACE_VERTEX_NORMAL_COMPUTE_ID,
		D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE),
	fluidMeshPipeline("FluidMeshShader.cso", "FluidSurfaceShader.cso", "FluidMeshRootSig.cso", *context, CommandListID::FLUID_MESH_ID,
		D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE),
	bufferClearCP("bufferClearRootSignature.cso", "bufferClearComputeShader.cso", *context, CommandListID::FLUID_BUFFER_CLEAR_COMPUTE_ID,
		D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE),
	fluidScene(context, &fluidRP, &bilevelUniformGridCP, &surfaceBlockDetectionCP, &surfaceCellDetectionCP, &surfaceVertexCompactionCP, &surfaceVertexDensityCP, &surfaceVertexNormalCP, &bufferClearCP, &fluidMeshPipeline),
	currentRP(),
	currentCP()
{}

RenderPipeline* Scene::getObjectPipeline() {
	return &objectRP;
}

RenderPipeline* Scene::getPBMPMRenderPipeline() {
	return &pbmpmRP;
}

MeshPipeline* Scene::getFluidMeshPipeline() {
	return &fluidMeshPipeline;
}

void Scene::compute() {
	pbmpmScene.compute();
	fluidScene.compute(
		pbmpmScene.getPositionBuffer(),
		pbmpmScene.getParticleCount()
	);
}

void Scene::drawPBMPM() {
	pbmpmScene.draw(camera);
}

void Scene::drawFluid(unsigned int renderMeshlets) {
	fluidScene.draw(camera, renderMeshlets);
}

void Scene::drawObject() {
	objectScene.draw(camera);
}

void Scene::releaseResources() {
	objectScene.releaseResources();
	pbmpmScene.releaseResources();
	fluidScene.releaseResources();
}

void Scene::updatePBMPMConstants(PBMPMConstants& newConstants) {
	pbmpmScene.updateConstants(newConstants);
}
