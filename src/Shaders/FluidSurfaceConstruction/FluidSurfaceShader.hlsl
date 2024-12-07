#include "FluidMeshRootSig.hlsl"
#include "../constants.h"

struct PSInput {
    float4 ndcPos: SV_Position;
    float3 normal: NORMAL0;
    float3 worldPos: POSITION1;
    int meshletIndex: COLOR0;
};

ConstantBuffer<MeshShadingConstants> cb : register(b0);

float3 radiance(float3 dir)
{
    // Paper uses a sky model for the radiance
    // Return a constant sky-like color
    return float3(0.53, 0.81, 0.92); // Light sky blue
}

float fresnelSchlick(float VdotH, float F0) {
    return F0 + (1.0 - F0) * pow(1.0 - VdotH, 5.0);
}

bool intersectAABB(in float3 origin, in float3 direction, 
                   in float3 aabbMin, in float3 aabbMax,
                   out float tMin, out float tMax)
{
    float3 t1 = (aabbMin - origin) / direction;
    float3 t2 = (aabbMax - origin) / direction;
    float3 min1 = min(t1, t2);
    float3 max1 = max(t1, t2);
    tMin = max(max(min1.x, min1.y), min1.z);
    tMax = min(min(max1.x, max1.y), max1.z);
    return 0 <= tMax && tMin <= tMax;
}

float remapFrom01(float value, float minValue, float maxValue) {
    return minValue + value * (maxValue - minValue);
}

float remapTo01(float value, float minValue, float maxValue) {
    return (value - minValue) / (maxValue - minValue);
}

float3 gammaCorrect(float3 color)
{
    float correctionFactor = (1.0 / 2.2);
    return pow(color, float3(correctionFactor, correctionFactor, correctionFactor));
}

float3 getMeshletColor(int index)
{
    float r = frac(sin(float(index) * 12.9898) * 43758.5453);
    float g = frac(sin(float(index) * 78.233) * 43758.5453);
    float b = frac(sin(float(index) * 43.853) * 43758.5453);
    return float3(r, g, b);
}

static const float3 baseColor = float3(0.7, 0.9, 1);

[RootSignature(ROOTSIG)]
float4 main(PSInput input) : SV_Target
{
    // refract
    float3 pos = input.worldPos;
    float3 dir = normalize(pos - cb.cameraPos);

    float ior = 1.33;
    float eta = 1.0 / ior;

    float3 reflectDir = reflect(dir, input.normal);
    float3 h = normalize(-dir + reflectDir);
    float3 reflection = radiance(reflectDir);
    float f0 = ((1.0 - ior) / (1.0 + ior)) * ((1.0 - ior) / (1.0 + ior));
    float fr = fresnelSchlick(dot(-dir, h), f0);

    float3 refractDir = refract(dir, input.normal, eta);
    float3 refraction;

    float tMin, tMax;
    if(intersectAABB(pos, refractDir, 
                    cb.minBounds, cb.minBounds + cb.resolution * cb.dimensions,
                    tMin, tMax)){
        float dist = tMax;
        float3 trans = clamp(exp(-remapTo01(dist, 1.0, 5.0)), 0.0, 1.0) * baseColor;
        refraction = trans * radiance(refractDir);
    }

    float3 baseColor = refraction * (1.0 - fr) + reflection * fr;
    return float4(gammaCorrect(baseColor), 0.5);
}