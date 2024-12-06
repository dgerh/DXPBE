#include "g2p2gRootSignature.hlsl"  // Includes the ROOTSIG definition
#include "../../pbmpmShaders/PBMPMCommon.hlsl"  // Includes the TileDataSize definition

// Taken from https://github.com/electronicarts/pbmpm

// Root constants bound to b0
ConstantBuffer<PBMPMConstants> g_simConstants : register(b0);

// Structured Buffer for particles (read-write UAV)
RWStructuredBuffer<Particle> g_particles : register(u0);

// Structured Buffer for free indices with atomic access (read-write UAV)
RWStructuredBuffer<int> g_freeIndices : register(u1);

// Structured Buffer for bukkit particle indices (read-only SRV)
StructuredBuffer<uint> g_bukkitParticleData : register(t0);

// Structured Buffer for bukkit thread data (read-only SRV)
StructuredBuffer<BukkitThreadData> g_bukkitThreadData : register(t1);

// Structured Buffer for grid source data (read-only SRV)
StructuredBuffer<int> g_gridSrc : register(t2);

// Structured Buffer for grid destination data (read-write UAV with atomic support)
RWStructuredBuffer<int> g_gridDst : register(u2);

// Structured Buffer for grid cells to be cleared (read-write UAV)
RWStructuredBuffer<int> g_gridToBeCleared : register(u3);

// Shared memory for tile data
RWStructuredBuffer<int> g_tempTileData : register(u4);

//groupshared int s_tileData[TileDataSize];
groupshared int s_tileDataDst[TileDataSize];

unsigned int localGridIndex(uint2 index) {
	return (index.y * TotalBukkitEdgeLength + index.x) * 4;
}

// Function to clamp a particle's position inside the guardian region of the grid
float2 projectInsideGuardian(float2 p, uint2 gridSize, float guardianSize)
{
    // Define the minimum and maximum clamp boundaries
    float2 clampMin = float2(guardianSize, guardianSize);
    float2 clampMax = float2(gridSize) - float2(guardianSize, guardianSize) - float2(1.0, 1.0);

    // Clamp the position `p` to be within the defined boundaries
    return clamp(p, clampMin, clampMax);
}

// Matrix Helper Functions

// Structure to hold the SVD result

// Function to compute the determinant of a 2x2 matrix
float det(float2x2 m)
{
    return m[0][0] * m[1][1] - m[0][1] * m[1][0];
}

// Function to compute the trace of a 2x2 matrix
float tr(float2x2 m)
{
    return m[0][0] + m[1][1];
}

// Function to compute the inverse of a 2x2 matrix
float2x2 inverse(float2x2 m)
{
    float a = m[0][0];
    float b = m[1][0];
    float c = m[0][1];
    float d = m[1][1];
    return (1.0 / det(m)) * float2x2(d, -c, -b, a);
}

// Function to compute the outer product of two float2 vectors
float2x2 outerProduct(float2 x, float2 y)
{
    return float2x2(x.x * y.x, x.x * y.y, x.y * y.x, x.y * y.y);
}

// Function to create a diagonal 2x2 matrix from a float2 vector
float2x2 diag(float2 d)
{
    return float2x2(d.x, 0, 0, d.y);
}

// Function to truncate 4x4 matrix to 2x2 matrix
float2x2 truncate(float4x4 m)
{
    return float2x2(m[0].xy, m[1].xy);
}

float4x4 expandToFloat4x4(float2x2 m)
{
    return float4x4(
        m[0][0], m[0][1], 0.0, 0.0,
        m[1][0], m[1][1], 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0
    );
}

struct SVDResult
{
    float2x2 U;
    float2 Sigma;
    float2x2 Vt;
};

// Function to compute SVD for a 2x2 matrix
SVDResult svd(float2x2 m)
{
    float E = (m[0][0] + m[1][1]) * 0.5;
    float F = (m[0][0] - m[1][1]) * 0.5;
    float G = (m[0][1] + m[1][0]) * 0.5;
    float H = (m[0][1] - m[1][0]) * 0.5;

    float Q = sqrt(E * E + H * H);
    float R = sqrt(F * F + G * G);
    float sx = Q + R;
    float sy = Q - R;

    float a1 = atan2(G, F);
    float a2 = atan2(H, E);

    float theta = (a2 - a1) * 0.5;
    float phi = (a2 + a1) * 0.5;

    float2x2 U = rot(phi);
    float2 Sigma = float2(sx, sy);
    float2x2 Vt = rot(theta);

    SVDResult result;
    result.U = U;
    result.Sigma = Sigma;
    result.Vt = Vt;

    return result;
}


[numthreads(ParticleDispatchSize, 1, 1)]
void main(uint indexInGroup : SV_GroupIndex, uint3 groupId : SV_GroupID)
{

    // Load thread-specific data
    BukkitThreadData threadData = g_bukkitThreadData[groupId.x];

    // Calculate grid origin
    int2 localGridOrigin = BukkitSize * int2(uint2(threadData.bukkitX, threadData.bukkitY)) - int2(BukkitHaloSize, BukkitHaloSize);
    int2 idInGroup = int2(int(indexInGroup) % TotalBukkitEdgeLength, int(indexInGroup) / TotalBukkitEdgeLength);
    int2 gridVertex = idInGroup + localGridOrigin;
    float2 gridPosition = float2(gridVertex);

    // Initialize variables
    float dx = 0.0;
    float dy = 0.0;
    float w = 0.0;
    float v = 0.0;

    // Check if grid vertex is within valid bounds
    bool gridVertexIsValid = all(gridVertex >= int2(0, 0)) && all(gridVertex <= g_simConstants.gridSize);

    if (gridVertexIsValid)
    {
        uint gridVertexAddress = gridVertexIndex(uint2(gridVertex), g_simConstants.gridSize);

		// Load grid data
        dx = decodeFixedPoint(g_gridSrc[gridVertexAddress + 0], g_simConstants.fixedPointMultiplier);
        dy = decodeFixedPoint(g_gridSrc[gridVertexAddress + 1], g_simConstants.fixedPointMultiplier);
        w = decodeFixedPoint(g_gridSrc[gridVertexAddress + 2], g_simConstants.fixedPointMultiplier);
        v = decodeFixedPoint(g_gridSrc[gridVertexAddress + 3], g_simConstants.fixedPointMultiplier);

        // Grid update
        if (w < 1e-5f)
        {
            dx = 0.0f;
            dy = 0.0f;
        }
        else
        {
            dx /= w;
            dy /= w;
        }

        float2 gridDisplacement = float2(dx, dy);

        // Collision detection against guardian shape

        // Grid vertices near or inside the guardian region should have their displacement values
        // corrected in order to prevent particles moving into the guardian.
        // We do this by finding whether a grid vertex would be inside the guardian region after displacement
        // with the current velocity and, if it is, setting the displacement so that no further penetration can occur.

        float2 displacedGridPosition = gridPosition + gridDisplacement;
        float2 projectedGridPosition = projectInsideGuardian(displacedGridPosition, g_simConstants.gridSize, GuardianSize + 1);
        float2 projectedDifference = projectedGridPosition - displacedGridPosition;

        if (projectedDifference.x != 0)
        {
            gridDisplacement.x = 0;
            gridDisplacement.y = lerp(gridDisplacement.y, 0, g_simConstants.borderFriction);
        }

        if (projectedDifference.y != 0)
        {
            gridDisplacement.y = 0;
            gridDisplacement.x = lerp(gridDisplacement.x, 0, g_simConstants.borderFriction);
        }

        dx = gridDisplacement.x;
        dy = gridDisplacement.y;
    }

    // Save grid to local memory
    unsigned int tileDataIndex = localGridIndex(idInGroup);
    // Store encoded fixed-point values atomically
    int originalValue;
    /*InterlockedExchange(s_tileData[tileDataIndex], encodeFixedPoint(dx, g_simConstants.fixedPointMultiplier), originalValue);
    InterlockedExchange(s_tileData[tileDataIndex + 1], encodeFixedPoint(dy, g_simConstants.fixedPointMultiplier), originalValue);
    InterlockedExchange(s_tileData[tileDataIndex + 2], encodeFixedPoint(w, g_simConstants.fixedPointMultiplier), originalValue);
    InterlockedExchange(s_tileData[tileDataIndex + 3], encodeFixedPoint(v, g_simConstants.fixedPointMultiplier), originalValue);*/
    InterlockedExchange(g_tempTileData[(groupId.x * TileDataSize) + tileDataIndex], encodeFixedPoint(dx, g_simConstants.fixedPointMultiplier), originalValue);
    InterlockedExchange(g_tempTileData[(groupId.x * TileDataSize) + tileDataIndex + 1], encodeFixedPoint(dy, g_simConstants.fixedPointMultiplier), originalValue);
    InterlockedExchange(g_tempTileData[(groupId.x * TileDataSize) + tileDataIndex + 2], encodeFixedPoint(w, g_simConstants.fixedPointMultiplier), originalValue);
    InterlockedExchange(g_tempTileData[(groupId.x * TileDataSize) + tileDataIndex + 3], encodeFixedPoint(v, g_simConstants.fixedPointMultiplier), originalValue);
    
    // Make sure all values in destination grid are 0
    InterlockedExchange(s_tileDataDst[tileDataIndex], 0, originalValue);
    InterlockedExchange(s_tileDataDst[tileDataIndex + 1], 0, originalValue);
    InterlockedExchange(s_tileDataDst[tileDataIndex + 2], 0, originalValue);
    InterlockedExchange(s_tileDataDst[tileDataIndex + 3], 0, originalValue);

    // Synchronize all threads in the group
    GroupMemoryBarrierWithGroupSync();
    
    if (indexInGroup < threadData.rangeCount)
    {
        // Load Particle
        uint myParticleIndex = g_bukkitParticleData[threadData.rangeStart + indexInGroup];
        
        Particle particle = g_particles[myParticleIndex];
        
        float2 p = particle.position;
        QuadraticWeightInfo weightInfo = quadraticWeightInit(p);
        
        if (g_simConstants.iteration != 0)
        {
            // G2P
            float2x2 B = ZeroMatrix;
            float2 d = float2(0, 0);
            float volume = 0.0;
            
            // Iterate over local 3x3 neighborhood
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    // Weight corresponding to this neighborhood cell
                    float weight = weightInfo.weights[i].x * weightInfo.weights[j].y;
                    
                    // Grid vertex index
                    int2 neighborCellIndex = int2(weightInfo.cellIndex) + int2(i, j);
                    
                    // 2D index relative to the corner of the local grid
                    int2 neighborCellIndexLocal = neighborCellIndex - localGridOrigin;
                    
                    // Linear Index in the local grid
                    uint gridVertexIdx = localGridIndex(uint2(neighborCellIndexLocal));
                    
                    int fixedPoint0;
                    //InterlockedAdd(s_tileData[gridVertexIdx + 0], 0, fixedPoint0);
                    InterlockedAdd(g_tempTileData[(groupId.x * TileDataSize) + gridVertexIdx + 0], 0, fixedPoint0);
                    int fixedPoint1;
                    //InterlockedAdd(s_tileData[gridVertexIdx + 1], 0, fixedPoint1);
                    InterlockedAdd(g_tempTileData[(groupId.x * TileDataSize) + gridVertexIdx + 1], 0, fixedPoint1);
                    
                    float2 weightedDisplacement = weight * float2(
                        decodeFixedPoint(fixedPoint0, g_simConstants.fixedPointMultiplier),
                        decodeFixedPoint(fixedPoint1, g_simConstants.fixedPointMultiplier));

                    float2 offset = float2(neighborCellIndex) - p + 0.5;
                    B += outerProduct(weightedDisplacement, offset);
                    d += weightedDisplacement;
                    
                    if (g_simConstants.useGridVolumeForLiquid != 0)
                    {
                        int fixedPoint3;
                        //InterlockedAdd(s_tileData[gridVertexIdx + 3], 0, fixedPoint3);
                        InterlockedAdd(g_tempTileData[(groupId.x * TileDataSize) + gridVertexIdx + 3], 0, fixedPoint3);
                        volume += weight * decodeFixedPoint(fixedPoint3, g_simConstants.fixedPointMultiplier);
                    }
                }

            }
            
            if (g_simConstants.useGridVolumeForLiquid != 0)
            {
                // Update particle volume
                
                volume = 1.0 / volume;
                if (volume < 1)
                {
                    particle.liquidDensity = lerp(particle.liquidDensity, volume, 0.1);
                }
            }
            
            // Save the deformation gradient as a 4x4 matrix by adding the identity matrix to the rest
            particle.deformationDisplacement = B * 4.0;
            particle.displacement = d;
            
            // Integration
            if (g_simConstants.iteration == g_simConstants.iterationCount - 1)
            {
                if (particle.material == MaterialLiquid)
                {
                    // The liquid material only cares about the determinant of the deformation gradient.
                    // We can use the regular MPM integration below to evolve the deformation gradient, but
                    // this approximation actually conserves more volume.
                    // This is based on det(F^n+1) = det((I + D)*F^n) = det(I+D)*det(F^n)
                    // and noticing that D is a small velocity, we can use the identity det(I + D) ≈ 1 + tr(D) to first order
                    // ending up with det(F^n+1) = (1+tr(D))*det(F^n)
                    // Then we directly set particle.liquidDensity to reflect the approximately integrated volume.
                    // The liquid material does not actually use the deformation gradient matrix.
                    particle.liquidDensity *= (tr(particle.deformationDisplacement) + 1.0);

                    // Safety clamp to avoid instability with very small densities.
                    particle.liquidDensity = max(particle.liquidDensity, 0.05);
                }
                else
                {
                    particle.deformationDisplacement = (Identity + particle.deformationDisplacement) * particle.deformationGradient;
                }
                
                // Update particle position
                particle.position += particle.displacement;
                
                // Mouse Iteraction Here
                if (g_simConstants.mouseActivation > 0)
                {
                    float2 offset = particle.position - g_simConstants.mousePosition;
                    float lenOffset = max(length(offset), 0.0001);
                    if (lenOffset < g_simConstants.mouseRadius)
                    {
                        float2 normOffset = offset / lenOffset;

                        if (g_simConstants.mouseFunction == 0)
                        {
                            particle.displacement += normOffset * 500.f;
                        }
                        else if (g_simConstants.mouseFunction == 1)
                        {
                            particle.displacement = g_simConstants.mouseVelocity * g_simConstants.deltaTime;
                        }
                    }
                }
                
                // Gravity Acceleration is normalized to the vertical size of the window
                particle.displacement.y -= float(g_simConstants.gridSize.y) * g_simConstants.gravityStrength * g_simConstants.deltaTime * g_simConstants.deltaTime;
                
                // Free count may be negative because of emission. So make sure it is at last zero before incrementing.
                int originalMax; // Needed for InterlockedMax output parameter
                InterlockedMax(g_freeIndices[0], 0, originalMax); 
                
                particle.position = projectInsideGuardian(particle.position, g_simConstants.gridSize, GuardianSize);
            }
            
            // Save the particle back to the buffer
            g_particles[myParticleIndex] = particle;        }
        
        {
            // Particle update
            if (particle.material == MaterialLiquid)
            {
                // Simple liquid viscosity: just remove deviatoric part of the deformation displacement
                float2x2 deviatoric = -1.0 * (particle.deformationDisplacement + transpose(particle.deformationDisplacement));
                particle.deformationDisplacement += g_simConstants.liquidViscosity * 0.5 * deviatoric;

                float alpha = 0.5 * (1.0 / particle.liquidDensity - tr(particle.deformationDisplacement) - 1.0);
                particle.deformationDisplacement += g_simConstants.liquidRelaxation * alpha * Identity;
            }

            // P2G

            // Iterate over local 3x3 neighborhood
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    // Weight corresponding to this neighborhood cell
                    float weight = weightInfo.weights[i].x * weightInfo.weights[j].y;

                    // Grid vertex index
                    int2 neighborCellIndex = int2(weightInfo.cellIndex) + int2(i, j);

                    // 2D index relative to the corner of the local grid
                    int2 neighborCellIndexLocal = neighborCellIndex - localGridOrigin;

                    // Linear Index in the local grid
                    uint gridVertexIdx = localGridIndex(uint2(neighborCellIndexLocal));

                    // Update grid data
                    float2 offset = float2(neighborCellIndex) - p + 0.5;

                    float weightedMass = weight * particle.mass;
                    float2 momentum = weightedMass * (particle.displacement + mul(particle.deformationDisplacement, offset));

                    InterlockedAdd(s_tileDataDst[gridVertexIdx + 0], encodeFixedPoint(momentum.x, g_simConstants.fixedPointMultiplier));
                    InterlockedAdd(s_tileDataDst[gridVertexIdx + 1], encodeFixedPoint(momentum.y, g_simConstants.fixedPointMultiplier));
                    InterlockedAdd(s_tileDataDst[gridVertexIdx + 2], encodeFixedPoint(weightedMass, g_simConstants.fixedPointMultiplier));


                    if (g_simConstants.useGridVolumeForLiquid != 0)
                    {
                        InterlockedAdd(s_tileDataDst[gridVertexIdx + 3], encodeFixedPoint(weight * particle.volume, g_simConstants.fixedPointMultiplier));
                    }
                }

            }
        }
    }
    
    // Synchronize all threads in the group
    GroupMemoryBarrierWithGroupSync();
    
    // Save Grid
    if (gridVertexIsValid)
    {
        uint gridVertexAddress = gridVertexIndex(uint2(gridVertex), g_simConstants.gridSize);

        // Atomic loads from shared memory using InterlockedAdd with 0

        int dxi, dyi, wi, vi;
        InterlockedAdd(s_tileDataDst[tileDataIndex + 0], 0, dxi);
        InterlockedAdd(s_tileDataDst[tileDataIndex + 1], 0, dyi);
        InterlockedAdd(s_tileDataDst[tileDataIndex + 2], 0, wi);
        InterlockedAdd(s_tileDataDst[tileDataIndex + 3], 0, vi);

    // Atomic adds to the destination buffer
        InterlockedAdd(g_gridDst[gridVertexAddress + 0], dxi);
        InterlockedAdd(g_gridDst[gridVertexAddress + 1], dyi);
        InterlockedAdd(g_gridDst[gridVertexAddress + 2], wi);
        InterlockedAdd(g_gridDst[gridVertexAddress + 3], vi);
    
    // Clear the entries in g_gridToBeCleared
        g_gridToBeCleared[gridVertexAddress + 0] = 0;
        g_gridToBeCleared[gridVertexAddress + 1] = 0;
        g_gridToBeCleared[gridVertexAddress + 2] = 0;
        g_gridToBeCleared[gridVertexAddress + 3] = 0;

		g_tempTileData[(groupId.x * TileDataSize) + tileDataIndex] = 0;
		g_tempTileData[(groupId.x * TileDataSize) + tileDataIndex + 1] = 0;
		g_tempTileData[(groupId.x * TileDataSize) + tileDataIndex + 2] = 0;
		g_tempTileData[(groupId.x * TileDataSize) + tileDataIndex + 3] = 0;
    }

}