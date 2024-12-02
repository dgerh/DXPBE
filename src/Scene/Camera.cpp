#include "Camera.h"

Camera::Camera() {
	setFOV(0.25f * XM_PI, float(SCREEN_WIDTH) / (float)SCREEN_HEIGHT, 0.1f, 1000.0f);
	updateProjMat();
	updateViewMat();
}

void Camera::setFOV(float p_FOVY, float p_aspect, float p_nearPlane, float p_farPlane) {
	FOVY = p_FOVY;
	aspect = p_aspect;
	nearPlane = p_nearPlane;
	farPlane = p_farPlane;
}

void Camera::updateAspect(float p_aspect) {
	aspect = p_aspect;
	updateProjMat();
}

void Camera::rotateOnX(float angle) {
	rotateX += angle;
}

void Camera::rotateOnY(float angle) {
	rotateY += angle;
}

void Camera::rotate() {
	// limit pitch to straight up or straight down
	constexpr float limit = XM_PIDIV2 - 0.01f;
	rotateX = std::max(-limit, rotateX);
	rotateX = std::min(+limit, rotateX);

	// keep longitude in sane range by wrapping
	if (rotateY > XM_PI)
	{
		rotateY -= XM_2PI;
	}
	else if (rotateY < -XM_PI)
	{
		rotateY += XM_2PI;
	}

	float y = sinf(rotateX);
	float r = cosf(rotateX);
	float z = r * cosf(rotateY);
	float x = r * sinf(rotateY);

	XMVECTOR newForward = XMLoadFloat3(&forward) + XMVECTOR{ x, y, z };

	XMVECTOR tempUp{ 0, 1, 0 };

	// Right vector is perpendicular to forward and up
	XMVECTOR newRight = XMVector3Cross(tempUp, newForward);
	newRight = XMVector3Normalize(newRight);

	// Up vector is perpendicular to forward and right
	XMVECTOR newUp = XMVector3Cross(newForward, newRight);
	newUp = XMVector3Normalize(newUp);

	// Store the updated vectors back into the class member variables
	XMStoreFloat3(&right, newRight);
	XMStoreFloat3(&up, newUp);
	XMStoreFloat3(&forward, newForward);
}

void Camera::translate(XMFLOAT3 distance) {
	XMVECTOR P = XMLoadFloat3(&position);
	XMVECTOR D = XMLoadFloat3(&distance);
	D *= MOVE_SCALAR;
	XMStoreFloat3(&position, XMVectorAdd(P, D));
}

void Camera::updateViewMat() {
	XMVECTOR R = XMLoadFloat3(&right);
	XMVECTOR U = XMLoadFloat3(&up);
	XMVECTOR F = XMLoadFloat3(&forward);
	XMVECTOR P = XMLoadFloat3(&position);

	F = XMVector3Normalize(F);
	U = XMVector3Normalize(XMVector3Cross(F, R));
	R = XMVector3Cross(U, F);

	float x = -XMVectorGetX(XMVector3Dot(P, R));
	float y = -XMVectorGetX(XMVector3Dot(P, U));
	float z = -XMVectorGetX(XMVector3Dot(P, F));

	XMStoreFloat3(&right, R);
	XMStoreFloat3(&up, U);
	XMStoreFloat3(&forward, F);

	viewMat(0, 0) = right.x;
	viewMat(1, 0) = right.y;
	viewMat(2, 0) = right.z;
	viewMat(3, 0) = -position.x;

	viewMat(0, 1) = up.x;
	viewMat(1, 1) = up.y;
	viewMat(2, 1) = up.z;
	viewMat(3, 1) = -position.y;

	viewMat(0, 2) = forward.x;
	viewMat(1, 2) = forward.y;
	viewMat(2, 2) = forward.z;
	viewMat(3, 2) = -position.z;

	viewMat(0, 3) = 0.0f;
	viewMat(1, 3) = 0.0f;
	viewMat(2, 3) = 0.0f;
	viewMat(3, 3) = 1.0f;

	XMStoreFloat4x4(&viewProjMat, XMLoadFloat4x4(&viewMat) * XMLoadFloat4x4(&projMat));
}

void Camera::updateProjMat() {
	XMMATRIX P = XMMatrixPerspectiveFovLH(FOVY, aspect, nearPlane, farPlane);
	XMStoreFloat4x4(&projMat, P);

	XMStoreFloat4x4(&viewProjMat, XMLoadFloat4x4(&viewMat) * XMLoadFloat4x4(&projMat));
}

XMMATRIX Camera::getViewMat() {
	return XMLoadFloat4x4(&viewMat);
}

XMMATRIX Camera::getProjMat() {
	return XMLoadFloat4x4(&projMat);
}

XMMATRIX Camera::getViewProjMat() {
	return XMLoadFloat4x4(&viewProjMat);
}