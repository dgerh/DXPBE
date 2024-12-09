#include "PBMPMRootSignature.hlsl"

cbuffer CameraMatrices : register(b0) {
    float4x4 viewMatrix;        // 16 floats
    float4x4 projectionMatrix;  // 16 floats
	float4x4 modelMatrix;       // 16 floats
};

// Particle positions as an SRV at register t0
StructuredBuffer<float4> positions : register(t0);

struct VSInput
{
    float3 Position : POSITION;      // Input position from vertex buffer
    uint InstanceID : SV_InstanceID; // Instance ID for indexing into model matrices
};

struct VSOutput
{
    float4 Position : SV_Position; // Position for rasterization
    uint InstanceID : INSTANCE_ID; // Pass the instance ID to the pixel shader
};

[RootSignature(ROOTSIG)]
VSOutput main(VSInput input)
{
    VSOutput output;

    // Retrieve the particle position for the current instance
    float3 particlePosition = positions[input.InstanceID].xyz;

    // Apply the model, view, and projection transformations
    float4 worldPos = mul(modelMatrix, float4(input.Position + particlePosition, 1.0));
    float4 viewPos = mul(viewMatrix, worldPos);
    output.Position = mul(projectionMatrix, viewPos);

    // Pass the instance ID to the pixel shader
    output.InstanceID = input.InstanceID;

    return output;
}