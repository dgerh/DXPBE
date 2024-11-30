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

groupshared int s_tileData[TileDataSize];
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

// Function to create a 2x2 rotation matrix
float2x2 rot(float theta)
{
    float ct = cos(theta);
    float st = sin(theta);

    return float2x2(ct, st, -st, ct);
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

// Define constants for identity and zero matrices
static const float2x2 Identity = float2x2(1, 0, 0, 1);
static const float2x2 ZeroMatrix = float2x2(0, 0, 0, 0);


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
    InterlockedExchange(s_tileData[tileDataIndex], encodeFixedPoint(dx, g_simConstants.fixedPointMultiplier), originalValue);
    InterlockedExchange(s_tileData[tileDataIndex + 1], encodeFixedPoint(dy, g_simConstants.fixedPointMultiplier), originalValue);
    InterlockedExchange(s_tileData[tileDataIndex + 2], encodeFixedPoint(w, g_simConstants.fixedPointMultiplier), originalValue);
    InterlockedExchange(s_tileData[tileDataIndex + 3], encodeFixedPoint(v, g_simConstants.fixedPointMultiplier), originalValue);
    
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
                    InterlockedAdd(s_tileData[gridVertexIdx + 0], 0, fixedPoint0);
                    int fixedPoint1;
                    InterlockedAdd(s_tileData[gridVertexIdx + 1], 0, fixedPoint1);
                    
                    float2 weightedDisplacement = weight * float2(
                        decodeFixedPoint(fixedPoint0, g_simConstants.fixedPointMultiplier),
                        decodeFixedPoint(fixedPoint1, g_simConstants.fixedPointMultiplier));

                    float2 offset = float2(neighborCellIndex) - p + 0.5;
                    B += outerProduct(weightedDisplacement, offset);
                    d += weightedDisplacement;
                    
                    if (g_simConstants.useGridVolumeForLiquid != 0)
                    {
                        int fixedPoint3;
                        InterlockedAdd(s_tileData[gridVertexIdx + 3], 0, fixedPoint3);
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
                if (particle.material != MaterialLiquid) {

                    SVDResult svdResult = svd(particle.deformationGradient);
                    // Safety clamp to prevent numerical instability
                    // Clamp each singular value to prevent extreme deformation
                    // 
                    //svdResult.Sigma = clamp(svdResult.Sigma, float3(0.1, 0.1, 0.1), float3(10000.0, 10000.0, 10000.0));
                    svdResult.Sigma = clamp(svdResult.Sigma, float2(0.1, 0.1), float2(10000.0, 10000.0));

                    if (particle.material == MaterialSand) {
                        // Drucker - Prager sand based on :
                        // Gergely Klár, Theodore Gast, Andre Pradhana, Chuyuan Fu, Craig Schroeder, Chenfanfu Jiang, and Joseph Teran. 2016.
                        // Drucker-prager elastoplasticity for sand animation. ACM Trans. Graph. 35, 4, Article 103 (July 2016), 12 pages.
                        // https://doi.org/10.1145/2897824.2925906
                        float sinPhi = sin(g_simConstants.frictionAngle * 3.14159 / 180.0);
                        float alpha = sqrt(2.0 / 3.0) * 2.0 * sinPhi / (3.0 - sinPhi);
                        float beta = 0.5;

                        //float3 eDiag = log(max(abs(svdResult.Sigma), float3(1e-6, 1e-6, 1e-6)));
                        //float3x3 eps = diag(eDiag);
                        //float trace = tr(eps) + particle.logJp;

                        //float3x3 eHat = eps - (trace / 3.0) * Identity3D;  // Note: Changed from 2 to 3 for 3D
                        //float frobNrm = sqrt(dot(eHat[0], eHat[0]) +
                        //    dot(eHat[1], eHat[1]) +
                        //    dot(eHat[2], eHat[2]));

                        float2 eDiag = log(max(abs(svdResult.Sigma), float2(1e-6, 1e-6)));
                        float2x2 eps = diag(eDiag);
                        float trace = tr(eps) + particle.logJp;

                        float2x2 eHat = eps - (trace / 2.0) * Identity;  // Note: Changed from 2 to 3 for 3D
                        float frobNrm = sqrt(dot(eHat[0], eHat[0]) +
                            dot(eHat[1], eHat[1]));

                        float elasticityRatio = 0.9f;
                        if (trace >= 0.0)
                        {
                            // Expansive motion - reset deformation
                            //svdResult.Sigma = float3(1.0, 1.0, 1.0);
                            svdResult.Sigma = float2(1.0, 1.0);
                            particle.logJp = beta * trace;
                        }
                        else
                        {
                            particle.logJp = 0;
                            float deltaGammaI = frobNrm + (elasticityRatio + 1.0) * trace * alpha;

                            if (deltaGammaI > 0)
                            {
                                // Project to yield surface
                                //float3 h = eDiag - deltaGammaI / frobNrm * (eDiag - float3(trace / 3.0, trace / 3.0, trace / 3.0));
                                float2 h = eDiag - deltaGammaI / frobNrm * (eDiag - (trace * 0.5));
                                svdResult.Sigma = exp(h);
                            }
                        }
                        particle.deformationGradient = mul(mul(svdResult.U, diag(svdResult.Sigma)), svdResult.Vt);
                    }

                    else if (particle.material == MaterialVisco)
                    {
                        float plasticity = 0.9f;
                        float yieldSurface = exp(1.0 - plasticity);

                        // Calculate current volume
                        //float J = svdResult.Sigma.x * svdResult.Sigma.y * svdResult.Sigma.z;  // Changed for 3D
                        float J = svdResult.Sigma.x * svdResult.Sigma.y;

                        /*svdResult.Sigma = clamp(svdResult.Sigma,
                            float3(1.0 / yieldSurface, 1.0 / yieldSurface, 1.0 / yieldSurface),
                            float3(yieldSurface, yieldSurface, yieldSurface));*/
                        svdResult.Sigma = clamp(svdResult.Sigma,
                            float2(1.0 / yieldSurface, 1.0 / yieldSurface),
                            float2(yieldSurface, yieldSurface));

                        //float newJ = svdResult.Sigma.x * svdResult.Sigma.y * svdResult.Sigma.z;
                        float newJ = svdResult.Sigma.x * svdResult.Sigma.y;
                        //svdResult.Sigma *= pow(J / newJ, 1.0 / 3.0);  // Changed for 3D: using cube root
                        svdResult.Sigma *= pow(J / newJ, 1.0 / 2.0);

                        particle.deformationGradient = mul(mul(svdResult.U, diag(svdResult.Sigma)), svdResult.Vt);
                    }
                     
                    else if (particle.material == MaterialSnow) {

                       

                        // Snow, snowCriticalCompression = 0.025, snowCriticalStretch = 0.0075,  snowHardeningCoeff = 10.0
                        //parameter and method reference based on A material point method for snow simulation
                        // Alexey Stomakhin, Craig Schroeder, Lawrence Chai, Joseph Teran, Andrew Selle 
                        //University of California Los Angeles and Walt Disney Animation Studios
                       //https://media.disneyanimation.com/uploads/production/publication_asset/94/asset/SSCTS13_2.pdf

                        SVDResult svdResult = svd(particle.deformationGradient);

                        // Snow parameters
                        float criticalCompression = 1.5e-2;
                        float criticalStretch = 7.5e-3;
                        float hardeningCoeff = 15.0;

                        // Elastic singular values clamping
                        float2 elasticSigma = clamp(svdResult.Sigma,
                            float2(1.0f - criticalCompression, 1.0f - criticalCompression),
                            float2(1.0f + criticalStretch, 1.0f + criticalStretch));

                        // Compute volume change from elastic part
                        float Je = elasticSigma.x * elasticSigma.y;

                        // Calculate hardening based on elastic volume change
                        float hardening = exp(hardeningCoeff * (1.0f - Je));

                        // Reconstruct elastic part Fe
                        float2x2 Fe = mul(mul(svdResult.U, diag(elasticSigma)), svdResult.Vt);

                        // Update plastic part Fp = F * Fe^(-1)
                        float2x2 FeInverse = mul(mul(svdResult.U, diag(1.0 / elasticSigma)), svdResult.Vt);
                        float2x2 Fp = mul(particle.deformationGradient, FeInverse);

                        // Update total deformation gradient F = Fe * Fp
                        particle.deformationGradient = mul(Fe * hardening, Fp);
                    }
                   
                }
                
                // Update particle position
                particle.position += particle.displacement;
                
                // Mouse Iteraction Here
                
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
            else if (particle.material == MaterialSand)
            {
                // Calculate deformation gradient F
                float2x2 F = mul(Identity + particle.deformationDisplacement, particle.deformationGradient);
                SVDResult svdResult = svd(F);

                if (particle.logJp == 0)
                {
                    svdResult.Sigma = clamp(svdResult.Sigma, float2(1, 1), float2(1000, 1000));
                }
                
                // Calculate closest matrix to F with det == 1
                float df = det(F);
                float cdf = clamp(abs(df), 0.1, 1.0);
                float2x2 Q = mul((1.0 / (sign(df) * sqrt(cdf))), F);

                float2x2 sigmaMat = diag(svdResult.Sigma);

                float elasticityRatio = 0.9f;
                float alpha = elasticityRatio;
                float2x2 tgt =  alpha * mul(mul(svdResult.U, sigmaMat), svdResult.Vt) + (1.0 - alpha) * Q;

                // Calculate and apply displacement difference
                float2x2 invDefGrad = inverse(particle.deformationGradient);
                float2x2 diff = mul(tgt, invDefGrad) - Identity - particle.deformationDisplacement;
                particle.deformationDisplacement += elasticityRatio * diff;

                // Apply viscosity
                float2x2 deviatoric = -1.0 * (particle.deformationDisplacement +
                    transpose(particle.deformationDisplacement));
                particle.deformationDisplacement += g_simConstants.liquidViscosity * 0.5 * deviatoric;
            }
            else if (particle.material == MaterialElastic || particle.material == MaterialVisco) {
                // Calculate total deformation gradient
                float2x2 F = mul(Identity + particle.deformationDisplacement, particle.deformationGradient);
                SVDResult svdResult = svd(F);

                float elasticRelaxation = 1.5f;
                float elasticityRatio = 0.1f;

                // Calculate matrix closest to F with determinant = 1 (volume preserving)
                float df = det(F);
                float cdf = clamp(abs(df), 0.1, 1000.0);
                float2x2 Q = mul((1.0 / (sign(df) * sqrt(cdf))), F);

                // Interpolate between rotation (svdResult.U * svdResult.Vt) and 
                // volume preserving (Q) target shapes
                float alpha = elasticityRatio;
                float2x2 rotationPart = mul(svdResult.U, svdResult.Vt);
                float2x2 targetState = alpha * rotationPart + (1.0 - alpha) * Q;

                // Calculate displacement difference
                float2x2 invDefGrad = inverse(particle.deformationGradient);
                float2x2 diff = mul(targetState, invDefGrad) - Identity - particle.deformationDisplacement;

                // Apply relaxation
                particle.deformationDisplacement += elasticRelaxation * diff;
            }
            else if (particle.material == MaterialSnow) {

                float2x2 F = mul(Identity + particle.deformationDisplacement, particle.deformationGradient);
                SVDResult svdResult = svd(F);

                // Use different parameters for displacement update to avoid overshooting
                float criticalCompression = 2.5e-2;  // Increased to allow more compression in displacement
                float criticalStretch = 5.0e-3;      // Reduced to limit stretching
                float hardeningCoeff = 10.0;         // Reduced to avoid double-hardening
                float snowViscosity = 0.1f;          // Increased for more stability
                float repulsionStrength = 0.15f;     // Adjusted for balance

                // Calculate elastic component
                float2 elasticSigma = clamp(svdResult.Sigma,
                    float2(1.0f - criticalCompression, 1.0f - criticalCompression),
                    float2(1.0f + criticalStretch, 1.0f + criticalStretch));

                float Je = elasticSigma.x * elasticSigma.y;

                // Modified hardening - use smaller coefficient since we already applied hardening
                float hardening = exp(hardeningCoeff * (1.0f - Je) * 0.5);

                // Improved compression handling
                if (Je < 0.85) { // Increased threshold
                    // Use quadratic falloff for smoother response
                    float compressionRatio = (0.85 - Je) / 0.85;
                    float repulsion = repulsionStrength * compressionRatio * compressionRatio;

                    // Add directional repulsion based on SVD
                    float2x2 repulsionDir = mul(mul(svdResult.U, Identity), svdResult.Vt);
                    particle.deformationDisplacement += repulsion * repulsionDir;
                }

                // Update displacement with modified elastic response
                float2x2 Fe = mul(mul(svdResult.U, diag(elasticSigma)), svdResult.Vt);
                float2x2 invF = inverse(F);
                float2x2 diff = mul(Fe * hardening, invF) - Identity - particle.deformationDisplacement;

                // Apply displacement update with relaxation
                float relaxationRate = 0.8f;  // Slower update for stability
                particle.deformationDisplacement += snowViscosity * diff * relaxationRate;

                // Enhanced viscosity handling
                float2x2 deviatoric = -1.0 * (particle.deformationDisplacement + transpose(particle.deformationDisplacement));

                // Add anisotropic damping
                float2x2 dampingMatrix = mul(mul(svdResult.U,
                    float2x2(1.2, 0,    // More damping in compression direction
                        0, 0.8)),   // Less damping in stretch direction
                    svdResult.Vt);

                particle.deformationDisplacement += snowViscosity * 0.5 * mul(deviatoric, dampingMatrix);

                // Add velocity damping based on deformation rate
                float deformationRate = length(particle.displacement);
                float velocityDamping = saturate(deformationRate * 2.0);
                particle.displacement *= (1.0 - velocityDamping * snowViscosity);
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
    }

}