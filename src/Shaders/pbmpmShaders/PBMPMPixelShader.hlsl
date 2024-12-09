#include "PBMPMRootSignature.hlsl"

// Particle materials as an SRV at register t1
StructuredBuffer<int> materials : register(t1);

struct PSInput
{
    float4 Position : SV_Position; // Position passed from vertex shader
    uint InstanceID : INSTANCE_ID; // Instance ID passed from vertex shader
};

[RootSignature(ROOTSIG)]
float4 main(PSInput input) : SV_Target
{
    // Use the instance ID to retrieve the material
    int materialType = materials[input.InstanceID];

// Determine the material color based on the material index (example logic)
if (materialType == 0)
    return float4(0.00, 0.63, 0.98, 1.0f); // Water
else if (materialType == 1)
    return float4(0.0f, 0.75f, 0.0f, 1.0f); // Elastic
else if (materialType == 2)
    return float4(0.8f, 0.8f, 0.0f, 1.0f); // Sand
else if (materialType == 3)
    return float4(0.7f, 0.0f, 0.8f, 1.0f); // Visco
else if (materialType == 4)
    return float4(0.8f, 0.8f, 0.8f, 1.0f); // Snow
else
    return float4(0.0f, 0.0f, 0.0f, 1.0f); // Default
}
