#pragma once

#include <filesystem>
#include <vector>

#include "Drawable.h"
#include "Mesh.h"
#include "PBMPMScene.h"

class ObjectScene : public Drawable {
public:
	ObjectScene(DXContext* context, RenderPipeline* pipeline, std::vector<SimShape>& shapes);

	void constructScene();

	void draw(Camera* camera);

	size_t getSceneSize();

	void releaseResources();

private:
	std::vector<std::string> inputStrings;
	std::vector<Mesh> meshes;
	std::vector<XMFLOAT4X4> modelMatrices;

	std::vector<SimShape> shapes;

	size_t sceneSize{ 0 };
};