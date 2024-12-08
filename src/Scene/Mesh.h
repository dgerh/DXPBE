#pragma once

#include <algorithm>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>

#include "Support/WinInclude.h"

#include "D3D/DXContext.h"
#include "D3D/VertexBuffer.h"
#include "D3D/IndexBuffer.h"
#include "D3D/StructuredBuffer.h"

#include "D3D/Pipeline/RenderPipeline.h"

using namespace DirectX;

struct Vertex {
	XMFLOAT3 pos;
	XMFLOAT3 nor;
	XMFLOAT3 col;
};

class Mesh {
public:
	Mesh() = delete;
	Mesh(std::string fileLocation, DXContext* context, ID3D12GraphicsCommandList6* cmdList, RenderPipeline* pipeline, XMFLOAT4X4 modelMatrix, bool doWireframeIndices = false);
	void loadMesh(std::string fileLocation, bool doWireframeIndices);

	D3D12_INDEX_BUFFER_VIEW* getIBV();
	D3D12_VERTEX_BUFFER_VIEW* getVBV();

	XMFLOAT4X4* getModelMatrix();

	void releaseResources();

	UINT getNumTriangles();

private:
	std::vector<Vertex> vertices;
	std::vector<XMFLOAT3> vertexPositions;
	std::vector<unsigned int> indices;
	UINT numTriangles;

	VertexBuffer vertexBuffer;
	IndexBuffer indexBuffer;
	D3D12_VERTEX_BUFFER_VIEW vbv;
	D3D12_INDEX_BUFFER_VIEW ibv;

	XMFLOAT4X4 modelMatrix;
};