#pragma once

#include "ObjectScene.h"
#include "PBMPMScene.h"
#include "FluidScene.h"
#include "../D3D/Pipeline/RenderPipeline.h"
#include "../D3D/Pipeline/ComputePipeline.h"
#include "../D3D/Pipeline/MeshPipeline.h"

enum RenderScene {
	Object,
	PBMPM,
	Physics,
	Fluid
};

class Scene {
public:
	Scene() = delete;
	Scene(RenderScene renderScene, Camera* camera, DXContext* context);

	RenderPipeline* getRenderPipeline();

	void setRenderScene(RenderScene renderScene);
	void compute();
	void draw();

	void releaseResources();

	void updatePBMPMConstants(PBMPMConstants& newConstants);

private:
	Camera* camera;

	RenderPipeline objectRP;
	ObjectScene objectScene;
	
	RenderPipeline pbmpmRP;
	unsigned int pbmpmIC;
	PBMPMScene pbmpmScene;

	RenderPipeline fluidRP;
	ComputePipeline bilevelUniformGridCP;
	ComputePipeline surfaceBlockDetectionCP;
	ComputePipeline surfaceCellDetectionCP;
	ComputePipeline surfaceVertexCompactionCP;
	ComputePipeline surfaceVertexDensityCP;
	ComputePipeline surfaceVertexNormalCP;
	MeshPipeline fluidMeshPipeline;
	FluidScene fluidScene;

	RenderScene scene;

	RenderPipeline* currentRP;
	ComputePipeline* currentCP;
};
