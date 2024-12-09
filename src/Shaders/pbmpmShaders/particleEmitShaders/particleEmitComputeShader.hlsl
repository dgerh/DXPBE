#include "particleEmitRootSignature.hlsl"  // Includes the ROOTSIG definition
#include "../../pbmpmShaders/PBMPMCommon.hlsl"  // Includes the TileDataSize definition

// Taken from https://github.com/electronicarts/pbmpm

// Root constants bound to b0
ConstantBuffer<PBMPMConstants> g_simConstants : register(b0);

// Define the constant buffer with an array of SimShapes
StructuredBuffer<SimShape> g_shapes : register(t0);

// Structured Buffer for particles (read-write UAV)
RWStructuredBuffer<Particle> g_particles : register(u0);

// Structured Buffer for free indices with atomic access (read-write UAV)
RWStructuredBuffer<int> g_freeIndices : register(u1);

// Structured Buffer for free indices with atomic access (read-write UAV)
RWStructuredBuffer<int> g_particleCount : register(u2);

// Structured Buffer for grid source data (read-only SRV)
StructuredBuffer<int> g_grid : register(t1);

// Structured Buffer for positions (read-write UAV)
RWStructuredBuffer<float4> g_positions : register(u3);

// Structured Buffer for materials (read-write UAV)
RWStructuredBuffer<int> g_materials : register(u4);

uint hash(uint input)
{
    uint state = input * 747796405 + 2891336453;
    uint word = ((state >> ((state >> 28) + 4)) ^ state) * 277803737;
    return (word >> 22) ^ word;
}

float randomFloat(uint input)
{
    return float(hash(input) % 10000) / 9999.0;
}

bool insideGuardian(uint3 id, uint3 gridSize, uint guardianSize)
{
    if (id.x <= guardianSize) { return false; }
    if (id.x >= (gridSize.x - guardianSize - 1)) { return false; }
    if (id.y <= guardianSize) { return false; }
    if (id.y >= gridSize.y - guardianSize - 1) { return false; }
    if (id.z <= guardianSize) { return false; }
    if (id.z >= gridSize.z - guardianSize - 1) { return false; }

    return true;
}

Particle createParticle(float mass, float volume)
{
    Particle particle;

    particle.displacement = float3(0, 0, 0);
    particle.deformationGradient = Identity;
    particle.deformationDisplacement = ZeroMatrix;

    particle.mass = mass;
    particle.volume = volume;

    particle.lambda = 0.0;
    particle.logJp = 1.0;
    particle.enabled = 1.0;

    return particle;
}

void addParticle(float3 position, int material, float volume, float density, float jitterScale)
{
    uint particleIndex = 0;
    // First check the free list to see if we can reuse a particle slot
    int freeIndexSlot;
    InterlockedAdd(g_freeIndices[0], -1, freeIndexSlot);
    freeIndexSlot--;

    if (freeIndexSlot >= 0)
    {
        InterlockedAdd(g_freeIndices[freeIndexSlot + 1], 0, particleIndex);
    }
    else // If free list is empty then grow the particle count
    {
        InterlockedAdd(g_particleCount[0], 1, particleIndex);
    }

    uint jitterX = hash(particleIndex);
    uint jitterY = hash(uint(position.x * position.y * 100));

    float2 jitter = float2(-0.25, -0.25) + 0.5 * float2(float(jitterX % 10) / 10, float(jitterY % 10) / 10);

    Particle newParticle = createParticle(
        volume * density,
        volume
    );

    g_particles[particleIndex] = newParticle;
    g_materials[particleIndex] = material;
    g_positions[particleIndex] = float4(position + float3(jitter.x, jitter.y, 0.f) * jitterScale, 1.0);
}

[numthreads(GridDispatchSize, GridDispatchSize, GridDispatchSize)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (!insideGuardian(id.xyz, g_simConstants.gridSize, GuardianSize + 1))
    {
        return;
    }

    float3 pos = float3(id.xyz);

    QuadraticWeightInfo weightInfo = quadraticWeightInit(pos);
    int3 nearestCell = int3(weightInfo.cellIndex) + int3(1, 1, 1);
    float nearestCellVolume = decodeFixedPoint(g_grid[gridVertexIndex(uint3(nearestCell), g_simConstants.gridSize) + 4], g_simConstants.fixedPointMultiplier);

    for (int shapeIndex = 0; shapeIndex < g_simConstants.shapeCount; shapeIndex++)
    {
        SimShape shape = g_shapes[shapeIndex];

        bool isEmitter = shape.functionality == ShapeFunctionEmit;
        bool isInitialEmitter = shape.functionality == ShapeFunctionInitialEmit;

        if (!(isEmitter || isInitialEmitter))
        {
            continue;
        }

        // Skip emission if we are spewing liquid into an already compressed space
        if (isEmitter && shape.material == MaterialLiquid && nearestCellVolume > 1.5)
        {
            continue;
        }

        uint particleCountPerCellAxis = uint(g_simConstants.particlesPerCellAxis);
        float volumePerParticle = 1.0f / float(particleCountPerCellAxis * particleCountPerCellAxis);

        CollideResult c = collide(shape, pos);
        if (c.collides)
        {
            uint emitEvery = uint(1.0 / (shape.emissionRate * g_simConstants.deltaTime));

            for (int i = 0; i < particleCountPerCellAxis; i++)
            {
                for (int j = 0; j < particleCountPerCellAxis; j++)
                {
                    uint hashCodeX = hash(id.x * particleCountPerCellAxis + i);
                    uint hashCodeY = hash(id.y * particleCountPerCellAxis + j);
                    uint hashCode = hash(hashCodeX + hashCodeY);

                    bool emitDueToMyTurnHappening = isEmitter && 0 == ((hashCode + g_simConstants.simFrame) % emitEvery);
                    bool emitDueToInitialEmission = isInitialEmitter && g_simConstants.simFrame == 0;

                    float3 emitPos = pos + float3(float(i), float(j), 0) / float(particleCountPerCellAxis);

                    if (emitDueToInitialEmission || emitDueToMyTurnHappening)
                    {
                        addParticle(emitPos, shape.material, volumePerParticle, 1.0, 1.0 / float(particleCountPerCellAxis));
                    }
                }
            }
        }
    }
}