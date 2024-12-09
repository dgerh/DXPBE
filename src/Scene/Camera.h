#pragma once

#include "Support/WinInclude.h"

#define SCREEN_WIDTH 1280
#define SCREEN_HEIGHT 720

#define MOVE_SCALAR 5.f

using namespace DirectX;

class Camera {
public:
	Camera();

	void setFOV(float FOVY, float aspect, float nearPlane, float farPlane);
	void updateAspect(float aspect);
	
	void rotateOnX(float angle);
	void rotateOnY(float angle);
	void rotate();
	void translate(XMFLOAT3 distance);

	void updateViewMat();

	XMFLOAT4 getForward() { return { forward.x, forward.y, forward.z, 0 }; }

	XMMATRIX getViewMat();
	XMMATRIX getProjMat();
	XMMATRIX getViewProjMat();
	XMFLOAT3 getPosition() { return position; }

	XMMATRIX getInvViewProjMat();
	
	void kmStateCheck(DirectX::Keyboard::State kState, DirectX::Mouse::State mState);

private:
	float FOVY;
	float aspect;
	float nearPlane;
	float farPlane;

	float rotateX{ 0 };
	float rotateY{ 0 };

	XMFLOAT3 position{ 30, 30, -60 };
	XMFLOAT3 up{ 0, 1, 0 };
	XMFLOAT3 forward{ 0, 0, 1 };
	XMFLOAT3 right{ 1, 0, 0 };

	XMFLOAT4X4 viewMat;
	XMFLOAT4X4 projMat;
	XMFLOAT4X4 viewProjMat;

	void updateProjMat();
};