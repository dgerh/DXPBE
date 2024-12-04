#include "particleEmitRootSignature.hlsl"  // Includes the ROOTSIG definition
#include "../../pbmpmShaders/PBMPMCommon.hlsl"  // Includes the TileDataSize definition

// Taken from https://github.com/electronicarts/pbmpm

// Root constants bound to b0
ConstantBuffer<PBMPMConstants> g_simConstants : register(b0);

// Define the constant buffer with an array of SimShapes
cbuffer shapes : register(b1)
{
    SimShape g_shapes[1]; // Adjust the size of the array as needed
};

// Structured Buffer for particles (read-write UAV)
RWStructuredBuffer<Particle> g_particles : register(u0);

// Structured Buffer for free indices with atomic access (read-write UAV)
RWStructuredBuffer<int> g_freeIndices : register(u1);

// Structured Buffer for free indices with atomic access (read-write UAV)
RWStructuredBuffer<int> g_particleCount : register(u2);

// Structured Buffer for grid source data (read-only SRV)
StructuredBuffer<int> g_grid : register(t0);

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

bool insideGuardian(uint2 id, uint2 gridSize, uint guardianSize)
{
    if (id.x <= guardianSize) { return false; }
    if (id.x >= (gridSize.x - guardianSize - 1)) { return false; }
    if (id.y <= guardianSize) { return false; }
    if (id.y >= gridSize.y - guardianSize - 1) { return false; }

    return true;
}

Particle createParticle(float2 position, float material, float mass, float volume)
{
    Particle particle;

    particle.position = position;
    particle.displacement = float2(0, 0);
    particle.deformationGradient = Identity;
    particle.deformationDisplacement = ZeroMatrix;

    particle.liquidDensity = 1.0;
    particle.mass = mass;
    particle.material = material;
    particle.volume = volume;

    particle.lambda = 0.0;
    particle.logJp = 1.0;
    particle.enabled = 1.0;

    return particle;
}

void addParticle(float2 position, float material, float volume, float density, float jitterScale)
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
        position + jitter * jitterScale,
        material,
        volume * density,
        volume
    );

    g_particles[particleIndex] = newParticle;
}

[numthreads(GridDispatchSize, GridDispatchSize, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (!insideGuardian(id.xy, g_simConstants.gridSize, GuardianSize + 1))
    {
        return;
    }

    uint2 gridSize = g_simConstants.gridSize;
    float2 pos = float2(id.xy);

    QuadraticWeightInfo weightInfo = quadraticWeightInit(pos);
    int2 nearestCell = int2(weightInfo.cellIndex) + int2(1, 1);
    float nearestCellVolume = decodeFixedPoint(g_grid[gridVertexIndex(uint2(nearestCell), g_simConstants.gridSize) + 3], g_simConstants.fixedPointMultiplier);

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

                    float2 emitPos = pos + float2(float(i), float(j)) / float(particleCountPerCellAxis);

                    if (emitDueToInitialEmission || emitDueToMyTurnHappening)
                    {
                        addParticle(emitPos, shape.material, volumePerParticle, 1.0, 1.0 / float(particleCountPerCellAxis));
                    }
                }
            }
        }
    }
}