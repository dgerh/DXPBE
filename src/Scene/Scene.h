#pragma once

#include "ObjectScene.h"
#include "PBMPMScene.h"
#include "FluidScene.h"
#include "../D3D/Pipeline/RenderPipeline.h"
#include "../D3D/Pipeline/ComputePipeline.h"
#include "../D3D/Pipeline/MeshPipeline.h"

class Scene {
public:
	Scene() = delete;
	Scene(Camera* camera, DXContext* context);

	RenderPipeline* getObjectPipeline();
	RenderPipeline* getPBMPMRenderPipeline();
	MeshPipeline* getFluidMeshPipeline();

	void compute();
	void drawPBMPM();
	void drawFluid(unsigned int renderMeshlets);
	void drawObject();

	void releaseResources();

	PBMPMConstants getPBMPMConstants() { return pbmpmScene.getConstants(); }
	void updatePBMPMConstants(PBMPMConstants& newConstants);

	float* getFluidIsovalue() { return fluidScene.getIsovalue(); }
	float* getFluidKernelScale() { return fluidScene.getKernelScale(); }

private:
	Camera* camera;

	RenderPipeline pbmpmRP;
	unsigned int pbmpmIC;
	PBMPMScene pbmpmScene;

	RenderPipeline objectRP;
	ObjectScene objectScene;

	RenderPipeline fluidRP;
	ComputePipeline bilevelUniformGridCP;
	ComputePipeline surfaceBlockDetectionCP;
	ComputePipeline surfaceCellDetectionCP;
	ComputePipeline surfaceVertexCompactionCP;
	ComputePipeline surfaceVertexDensityCP;
	ComputePipeline surfaceVertexNormalCP;
	ComputePipeline bufferClearCP;
	MeshPipeline fluidMeshPipeline;
	FluidScene fluidScene;

	RenderPipeline* currentRP;
	ComputePipeline* currentCP;
};
