#pragma once

#include "../D3D/DXContext.h"
#include "Drawable.h"
#include "../D3D/StructuredBuffer.h"
#include "../D3D/VertexBuffer.h"
#include "../D3D/IndexBuffer.h"
#include "../D3D/Pipeline/ComputePipeline.h"
#include "Geometry.h"
#include <iostream>
#include <math.h>

// Keep consistent with PBMPMCommon.hlsl

const unsigned int GridDispatchSize = 8;
const unsigned int BukkitSize = 2;
const unsigned int BukkitHaloSize = 1;

const float PARTICLE_RADIUS = 0.5f;

const unsigned int maxParticles = 500000;
const unsigned int maxTimestampCount = 2048;

struct PBMPMConstants {
	XMUINT3 gridSize; //2 -> 3
	float deltaTime;
	float gravityStrength;

	float liquidRelaxation;
	float liquidViscosity;
	unsigned int fixedPointMultiplier;

	unsigned int useGridVolumeForLiquid;
	unsigned int particlesPerCellAxis;

	float frictionAngle;
	unsigned int shapeCount;
	unsigned int simFrame;

	unsigned int bukkitCount;
	unsigned int bukkitCountX;
	unsigned int bukkitCountY;
	unsigned int bukkitCountZ; //added
	unsigned int iteration;
	unsigned int iterationCount;
	float borderFriction;
	float elasticRelaxation;
	float elasticityRatio;


	//mouse stuff
	XMFLOAT4 mousePosition;
	XMFLOAT4 mouseDirection;
	unsigned int mouseActivation;
	unsigned int mouseRadius;
	unsigned int mouseFunction;
	float mouseVelocity;
};

struct SimShape {
	int id;
	XMFLOAT3 position;
	float rotation;
	XMFLOAT3 halfSize;

	int shapeType;
	int functionality;
	int material;
	float emissionRate;
	int radius;
	XMFLOAT3 padding;
};

struct PBMPMParticle {
	XMFLOAT3 displacement; //2->3
	float mass;
	XMFLOAT3X3 deformationGradient;
	float volume;
	float lambda;
	XMFLOAT3X3 deformationDisplacement;
	float logJp;
	float enabled;
};

struct BukkitSystem {
	unsigned int countX;
	unsigned int countY;
	unsigned int countZ; //added Z
	unsigned int count;
	StructuredBuffer countBuffer;
	StructuredBuffer countBuffer2;
	StructuredBuffer particleData;
	StructuredBuffer threadData;
	StructuredBuffer dispatch;
	StructuredBuffer blankDispatch;
	StructuredBuffer particleAllocator;
	StructuredBuffer indexStart;
};

struct BukkitThreadData {
	unsigned int rangeStart;
	unsigned int rangeCount;
	unsigned int bukkitX;
	unsigned int bukkitY;
	unsigned int bukkitZ; //added Z
};

class PBMPMScene : public Drawable {
public:
	PBMPMScene(DXContext* context, RenderPipeline* renderPipeline);

	void constructScene();

	void compute();

	void draw(Camera* camera);

	void releaseResources();

	void updateConstants(PBMPMConstants& newConstants);

	static bool constantsEqual(PBMPMConstants& one, PBMPMConstants& two);

	StructuredBuffer* getPositionBuffer() { return &positionBuffer; }

	int transferAndGetNumParticles();
	unsigned int getNumParticles() { return numParticles; }

	PBMPMConstants getConstants() { return constants; }

	std::vector<SimShape>& getSimShapes() { return shapes; }

	unsigned int* getSubstepCount() { return &substepCount; }

private:
	DXContext* context;
	RenderPipeline* renderPipeline;

	ComputePipeline g2p2gPipeline;
	ComputePipeline bukkitCountPipeline;
	ComputePipeline bukkitAllocatePipeline;
	ComputePipeline bukkitInsertPipeline;
	ComputePipeline bufferClearPipeline;
	ComputePipeline emissionPipeline;
	ComputePipeline setIndirectArgsPipeline;

	PBMPMConstants constants;
	BukkitSystem bukkitSystem;

	XMMATRIX modelMat;
	D3D12_VERTEX_BUFFER_VIEW vbv;
	D3D12_INDEX_BUFFER_VIEW ibv;
	VertexBuffer vertexBuffer;
	IndexBuffer indexBuffer;
	ID3D12CommandSignature* commandSignature = nullptr;
	ID3D12CommandSignature* renderCommandSignature = nullptr;
	UINT64 fenceValue = 1;
	ComPointer<ID3D12Fence> fence;
	unsigned int indexCount = 0;

	unsigned int substepIndex = 0;

	// Particle Buffers
	StructuredBuffer positionBuffer;
	StructuredBuffer materialBuffer;

	// Scene Buffers
	StructuredBuffer particleBuffer;
	StructuredBuffer particleFreeIndicesBuffer;
	StructuredBuffer particleCount;
	StructuredBuffer particleSimDispatch;
	StructuredBuffer renderDispatchBuffer;
	StructuredBuffer shapeBuffer;
	StructuredBuffer tempTileDataBuffer;

	std::array<StructuredBuffer, 3> gridBuffers;

	std::vector<SimShape> shapes;

	void createBukkitSystem();

	void updateSimUniforms(unsigned int iteration);

	void resetBuffers(bool resetGrids = false);

	void bukkitizeParticles();

	void doEmission(StructuredBuffer* gridBuffer);

	unsigned int frameCount{ 0 };
	unsigned int startTime{ 0 };
	unsigned int endTime{ 0 };

	unsigned int substepCount{ 5 };
	unsigned int numParticles{ 0 };
};