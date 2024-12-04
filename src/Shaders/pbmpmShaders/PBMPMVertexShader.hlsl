#include "PBMPMVertexRootSignature.hlsl"

cbuffer CameraMatrices : register(b0) {
    float4x4 viewMatrix;        // 16 floats
    float4x4 projectionMatrix;  // 16 floats
	float4x4 modelMatrix;       // 16 floats
};

struct Particle
{
	float3 position; //2->3
	float liquidDensity;
	float3 displacement; //2->3
	float mass;
	float material;
	float volume;
	float lambda;
	float logJp;
	float enabled;
	float4x4 deformationGradient;
	float4x4 deformationDisplacement;
};

// Particle positions as an SRV at register t0
StructuredBuffer<Particle> particles : register(t0);

struct VSInput
{
    float3 Position : POSITION;      // Input position from vertex buffer
    uint InstanceID : SV_InstanceID; // Instance ID for indexing into model matrices
};

[RootSignature(ROOTSIG)]
float4 main(VSInput input) : SV_Position
{

    float3 particlePosition = float3(particles[input.InstanceID].position, 0);

    // Apply the model, view, and projection transformations
    float4 worldPos = mul(modelMatrix, float4(input.Position + particlePosition, 1.0));
    float4 viewPos = mul(viewMatrix, worldPos);
    return mul(projectionMatrix, viewPos);
}