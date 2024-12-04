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

unsigned int localGridIndex(uint3 index) {
	return (index.z * TotalBukkitEdgeLength * TotalBukkitEdgeLength + index.y * TotalBukkitEdgeLength + index.x) * 5;
}

// Function to clamp a particle's position inside the guardian region of the grid
float3 projectInsideGuardian(float3 p, uint3 gridSize, float guardianSize)
{
    // Define the minimum and maximum clamp boundaries
    float3 clampMin = float3(guardianSize + 1.0, guardianSize + 1.0, guardianSize + 1.0);
    float3 clampMax = float3(gridSize) - float3(guardianSize - 1.0, guardianSize - 1.0, guardianSize - 1.0);
    // Clamp the position `p` to be within the defined boundaries
    return clamp(p, clampMin, clampMax);
}

// Matrix Helper Functions

// Structure to hold the SVD result

// Function to compute the determinant of a  3x3 matrix
float det(float3x3 m) {
    return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
}

// Function to compute the trace of a 2x2 matrix
float tr(float2x2 m)
{
    return m[0][0] + m[1][1];
}


// Function to compute the trace of a 3x3 matrix
float tr3D(float3x3 m) {
    return m[0][0] + m[1][1] + m[2][2];
}

float3x3 rotX(float theta)
{
    float ct = cos(theta);
    float st = sin(theta);
    return float3x3(
        1, 0, 0,
        0, ct, -st,
        0, st, ct
    );
}
float3x3 rotY(float theta)
{
    float ct = cos(theta);
    float st = sin(theta);
    return float3x3(
        ct, 0, st,
        0, 1, 0,
        -st, 0, ct
    );
}
float3x3 rotZ(float theta)
{
    float ct = cos(theta);
    float st = sin(theta);
    return float3x3(
        ct, -st, 0,
        st, ct, 0,
        0, 0, 1
    );
}
// Function to create a rotation matrix from Euler angles
float3x3 rot3D(float3 angles) // angles in radians (x, y, z)
{
    // Compose rotations in ZYX order
    return mul(rotZ(angles.z), mul(rotY(angles.y), rotX(angles.x)));
}



// Function to compute the inverse of a 3x3 matrix
float3x3 inverse(float3x3 m) {
    float d = det(m);
    float3x3 adj;
    adj[0][0] = +(m[1][1] * m[2][2] - m[2][1] * m[1][2]);
    adj[0][1] = -(m[0][1] * m[2][2] - m[2][1] * m[0][2]);
    adj[0][2] = +(m[0][1] * m[1][2] - m[1][1] * m[0][2]);
    adj[1][0] = -(m[1][0] * m[2][2] - m[2][0] * m[1][2]);
    adj[1][1] = +(m[0][0] * m[2][2] - m[2][0] * m[0][2]);
    adj[1][2] = -(m[0][0] * m[1][2] - m[1][0] * m[0][2]);
    adj[2][0] = +(m[1][0] * m[2][1] - m[2][0] * m[1][1]);
    adj[2][1] = -(m[0][0] * m[2][1] - m[2][0] * m[0][1]);
    adj[2][2] = +(m[0][0] * m[1][1] - m[1][0] * m[0][1]);
    return adj * (1.0 / d);

}


float3x3 outerProduct(float3 x, float3 y) {
    return float3x3(
        x.x * y.x, x.x * y.y, x.x * y.z,
        x.y * y.x, x.y * y.y, x.y * y.z,
        x.z * y.x, x.z * y.y, x.z * y.z
    );
}

// Function to create a diagonal 3x3 matrix from a float3 vector
float3x3 diag(float3 d)
{
    return float3x3(
        d.x, 0, 0,
        0, d.y, 0,
        0, 0, d.z
    );
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

float4x4 expandToFloat4x4(float3x3 m)
{
    return float4x4(
        m[0][0], m[0][1], m[0][2], 0.0,
        m[1][0], m[1][1], m[1][2], 0.0,
        m[2][0], m[2][1], m[2][2], 0.0,
        0.0, 0.0, 0.0, 0.0
    );
}



struct SVDResult
{
    float3x3 U;
    float3 Sigma;
    float3x3 Vt;
};

float3x3 givensRotation(float c, float s, int i, int j, int n) {
    float3x3 G = Identity;
    G[i][i] = c;
    G[i][j] = -s;
    G[j][i] = s;
    G[j][j] = c;
    return G;
}

// Compute off-diagonal sum of squares
float offDiagonalSum(float3x3 mat) {
    float sum = 0.0f;
    for (int i = 0; i < 3; i++) {
        for (int j = i + 1; j < 3; j++) {
            sum += mat[i][j] * mat[i][j] + mat[j][i] * mat[j][i];
        }
    }
    return sum;
}

// Function to compute SVD for a 3x3 matrix
SVDResult svd(float3x3 A) {
    SVDResult result;
    float3x3 U = Identity;
    float3x3 V = Identity;
    float3x3 At = transpose(A);
    float3x3 AtA = mul(At, A);
    const int MAX_ITERATIONS = 8;
    const float EPSILON = 1e-6;
    // Jacobi iteration
    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        float offDiag = offDiagonalSum(AtA);
        if (offDiag < EPSILON) break;
        // Process each upper triangular element
        for (int p = 0; p < 3; p++) {
            for (int q = p + 1; q < 3; q++) {
                float pp = AtA[p][p];
                float qq = AtA[q][q];
                float pq = AtA[p][q];
                // Compute Jacobi rotation
                float theta = 0.5f * atan2(2.0f * pq, pp - qq);
                float c = cos(theta);
                float s = sin(theta);
                // Apply rotation
                float3x3 J = givensRotation(c, s, p, q, 3);
                float3x3 Jt = transpose(J);
                AtA = mul(mul(Jt, AtA), J);
                V = mul(V, J);
            }
        }
    }
    // Extract singular values and ensure they're positive
    float3 singularValues;
    for (int i = 0; i < 3; i++) {
        singularValues[i] = sqrt(max(AtA[i][i], 0.0f));
    }
    // Sort singular values in descending order and adjust matrices
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2 - i; j++) {
            if (singularValues[j] < singularValues[j + 1]) {
                // Swap singular values
                float temp = singularValues[j];
                singularValues[j] = singularValues[j + 1];
                singularValues[j + 1] = temp;
                // Swap columns in V
                float3 tempCol = float3(V[0][j], V[1][j], V[2][j]);
                V[0][j] = V[0][j + 1];
                V[1][j] = V[1][j + 1];
                V[2][j] = V[2][j + 1];
                V[0][j + 1] = tempCol.x;
                V[1][j + 1] = tempCol.y;
                V[2][j + 1] = tempCol.z;
            }
        }
    }

    // Compute U = AV/Sigma
    float3x3 Vt = transpose(V);
    U = float3x3(0, 0, 0, 0, 0, 0, 0, 0, 0);
    for (int i = 0; i < 3; i++) {
        float sigma = singularValues[i];
        if (sigma > EPSILON) {
            float3 col = float3(0, 0, 0);
            for (int j = 0; j < 3; j++) {
                col.x += A[0][j] * V[j][i];
                col.y += A[1][j] * V[j][i];
                col.z += A[2][j] * V[j][i];
            }
            float invSigma = 1.0f / sigma;
            U[0][i] = col.x * invSigma;
            U[1][i] = col.y * invSigma;
            U[2][i] = col.z * invSigma;
        }
    }

    // Ensure right-handed coordinate system
    float det_U = det(U);
    float det_V = det(V);
    if (det_U * det_V < 0) {
        U[0][2] = -U[0][2];
        U[1][2] = -U[1][2];
        U[2][2] = -U[2][2];
        singularValues[2] = -singularValues[2];
    }

    result.U = U;
    result.Sigma = singularValues;
    result.Vt = Vt;
    return result;
}


[numthreads(ParticleDispatchSize, 1, 1)]
void main(uint indexInGroup : SV_GroupIndex, uint3 groupId : SV_GroupID)
{
    // Load thread-specific data
    BukkitThreadData threadData = g_bukkitThreadData[groupId.x];

    // Calculate grid origin
    int3 localGridOrigin = BukkitSize * int3(threadData.bukkitX, threadData.bukkitY, threadData.bukkitZ)
        - int3(BukkitHaloSize, BukkitHaloSize, BukkitHaloSize);
    int3 idInGroup = int3(
        int(indexInGroup) % TotalBukkitEdgeLength,
        (int(indexInGroup) / TotalBukkitEdgeLength) % TotalBukkitEdgeLength,
        int(indexInGroup) / (TotalBukkitEdgeLength * TotalBukkitEdgeLength)
    );
    int3 gridVertex = idInGroup + localGridOrigin;
    float3 gridPosition = float3(gridVertex);

    // Initialize variables
    float dx = 0.0;
    float dy = 0.0;
    float dz = 0.0;
    float w = 0.0; //weight
    float v = 0.0; //volume

    // Check if grid vertex is within valid bounds
    bool gridVertexIsValid = all(gridVertex >= int3(0, 0, 0)) && all(gridVertex <= g_simConstants.gridSize);

    if (gridVertexIsValid)
    {
        uint gridVertexAddress = gridVertexIndex(uint3(gridVertex), g_simConstants.gridSize);

		// Load grid data
        dx = decodeFixedPoint(g_gridSrc[gridVertexAddress + 0], g_simConstants.fixedPointMultiplier);
        dy = decodeFixedPoint(g_gridSrc[gridVertexAddress + 1], g_simConstants.fixedPointMultiplier);
        dz = decodeFixedPoint(g_gridSrc[gridVertexAddress + 2], g_simConstants.fixedPointMultiplier);
        w = decodeFixedPoint(g_gridSrc[gridVertexAddress + 3], g_simConstants.fixedPointMultiplier);
        v = decodeFixedPoint(g_gridSrc[gridVertexAddress + 4], g_simConstants.fixedPointMultiplier);

        // Grid update
        if (w < 1e-5f)
        {
            dx = 0.0f;
            dy = 0.0f;
            dz = 0.0f;
        }
        else
        {
            dx /= w;
            dy /= w;
            dz /= w;
        }

        float3 gridDisplacement = float3(dx, dy, dz);

        // Collision detection against guardian shape

        // Grid vertices near or inside the guardian region should have their displacement values
        // corrected in order to prevent particles moving into the guardian.
        // We do this by finding whether a grid vertex would be inside the guardian region after displacement
        // with the current velocity and, if it is, setting the displacement so that no further penetration can occur.

        float3 displacedGridPosition = gridPosition + gridDisplacement;
        float3 projectedGridPosition = projectInsideGuardian(displacedGridPosition, g_simConstants.gridSize, GuardianSize + 1);
        float3 projectedDifference = projectedGridPosition - displacedGridPosition;

        if (any(projectedDifference != 0))
        {
            // Calculate normal direction
            float3 normal = normalize(projectedDifference);
            // Project out the normal component
            float3 tangential = gridDisplacement - normal * dot(gridDisplacement, normal);
            // Apply friction to tangential component
            gridDisplacement = tangential * (1.0 - g_simConstants.borderFriction);
        }

        dx = gridDisplacement.x;
        dy = gridDisplacement.y;
        dz = gridDisplacement.z;
    }

    // Save grid to local memory
    unsigned int tileDataIndex = localGridIndex(idInGroup);
    // Store encoded fixed-point values atomically
    int originalValue;
    InterlockedExchange(s_tileData[tileDataIndex], encodeFixedPoint(dx, g_simConstants.fixedPointMultiplier), originalValue);
    InterlockedExchange(s_tileData[tileDataIndex + 1], encodeFixedPoint(dy, g_simConstants.fixedPointMultiplier), originalValue);
    InterlockedExchange(s_tileData[tileDataIndex + 2], encodeFixedPoint(dz, g_simConstants.fixedPointMultiplier), originalValue);
    InterlockedExchange(s_tileData[tileDataIndex + 3], encodeFixedPoint(w, g_simConstants.fixedPointMultiplier), originalValue);
    InterlockedExchange(s_tileData[tileDataIndex + 4], encodeFixedPoint(v, g_simConstants.fixedPointMultiplier), originalValue);
    
    // Make sure all values in destination grid are 0
    InterlockedExchange(s_tileDataDst[tileDataIndex], 0, originalValue);
    InterlockedExchange(s_tileDataDst[tileDataIndex + 1], 0, originalValue);
    InterlockedExchange(s_tileDataDst[tileDataIndex + 2], 0, originalValue);
    InterlockedExchange(s_tileDataDst[tileDataIndex + 3], 0, originalValue);
    InterlockedExchange(s_tileDataDst[tileDataIndex + 4], 0, originalValue);

    // Synchronize all threads in the group
    GroupMemoryBarrierWithGroupSync();
    
    if (indexInGroup < threadData.rangeCount)
    {
        // Load Particle
        uint myParticleIndex = g_bukkitParticleData[threadData.rangeStart + indexInGroup];
        
        Particle particle = g_particles[myParticleIndex];
        
        float3 p = particle.position;
        QuadraticWeightInfo weightInfo = quadraticWeightInit(p);
        
        if (g_simConstants.iteration != 0)
        {
            // G2P
            float3x3 B = ZeroMatrix;
            float3 d = float3(0, 0, 0);
            float volume = 0.0;
            
            // Iterate over local 3x3 neighborhood
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    for (int k = 0; k < 3; k++) {
                        // Weight corresponding to this neighborhood cell
                        float weight = weightInfo.weights[i].x * weightInfo.weights[j].y * weightInfo.weights[k].z;

                        // Grid vertex index
                        int3 neighborCellIndex = int3(weightInfo.cellIndex) + int3(i, j, k);

                        // 3D index relative to the corner of the local grid
                        int3 neighborCellIndexLocal = neighborCellIndex - localGridOrigin;

                        // Linear Index in the local grid
                        uint gridVertexIdx = localGridIndex(uint3(neighborCellIndexLocal));

                        int fixedPoint0;
                        InterlockedAdd(s_tileData[gridVertexIdx + 0], 0, fixedPoint0);
                        int fixedPoint1;
                        InterlockedAdd(s_tileData[gridVertexIdx + 1], 0, fixedPoint1);
                        int fixedPoint2;
                        InterlockedAdd(s_tileData[gridVertexIdx + 2], 0, fixedPoint2);

                        float3 weightedDisplacement = weight * float3(
                            decodeFixedPoint(fixedPoint0, g_simConstants.fixedPointMultiplier),
                            decodeFixedPoint(fixedPoint1, g_simConstants.fixedPointMultiplier),
                            decodeFixedPoint(fixedPoint2, g_simConstants.fixedPointMultiplier));

                        float3 offset = float3(neighborCellIndex) - p + 0.5;
                        B += outerProduct(weightedDisplacement, offset);
                       
                        d += weightedDisplacement;
                        if (g_simConstants.useGridVolumeForLiquid != 0)
                        {
                            int fixedPoint4;
                            InterlockedAdd(s_tileData[gridVertexIdx + 3], 0, fixedPoint4);
                            volume += weight * decodeFixedPoint(fixedPoint4, g_simConstants.fixedPointMultiplier);
                        }
                    }
                }

            }
            
            if (g_simConstants.useGridVolumeForLiquid != 0)
            {
                // Update particle volume
                
                volume = 1.0 / volume;
                if (volume < 1.0)
                {
                    particle.liquidDensity = lerp(particle.liquidDensity, volume, 0.1);
                }
            }
            
            // Save the deformation gradient as a 4x4 matrix by adding the identity matrix to the rest
            particle.deformationDisplacement = expandToFloat4x4(B) * 4.0;
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
                    particle.liquidDensity *= (tr3D(particle.deformationDisplacement) + 1.0);

                    // Safety clamp to avoid instability with very small densities.
                    particle.liquidDensity = max(particle.liquidDensity, 0.05);

                 
                }
                else
                {
                    particle.deformationDisplacement = expandToFloat4x4((Identity + particle.deformationDisplacement) * particle.deformationGradient);
                }
                if (particle.material != MaterialLiquid) {

                    SVDResult svdResult = svd(particle.deformationGradient);
                    // Safety clamp to prevent numerical instability
                    // Clamp each singular value to prevent extreme deformation
                    // 
                    svdResult.Sigma = clamp(svdResult.Sigma, float3(0.1, 0.1, 0.1), float3(10000.0, 10000.0, 10000.0));

                    if (particle.material == MaterialSand) {
                        // Drucker - Prager sand based on :
                        // Gergely Klár, Theodore Gast, Andre Pradhana, Chuyuan Fu, Craig Schroeder, Chenfanfu Jiang, and Joseph Teran. 2016.
                        // Drucker-prager elastoplasticity for sand animation. ACM Trans. Graph. 35, 4, Article 103 (July 2016), 12 pages.
                        // https://doi.org/10.1145/2897824.2925906
                        float sinPhi = sin(g_simConstants.frictionAngle * 3.14159 / 180.0);
                        float alpha = sqrt(2.0 / 3.0) * 2.0 * sinPhi / (3.0 - sinPhi);
                        float beta = 0.5;

                        float3 eDiag = log(max(abs(svdResult.Sigma), float3(1e-6, 1e-6, 1e-6)));
                        float3x3 eps = diag(eDiag);
                        float trace = tr3D(eps) + particle.logJp;

                        float3x3 eHat = eps - (trace / 3.0) * Identity;  // Note: Changed from 2 to 3 for 3D
                        float frobNrm = sqrt(dot(eHat[0], eHat[0]) +
                            dot(eHat[1], eHat[1]) +
                            dot(eHat[2], eHat[2]));

                        

                        float elasticityRatio = 0.9f;
                        if (trace >= 0.0)
                        {
                            // Expansive motion - reset deformation
                            svdResult.Sigma = float3(1.0, 1.0, 1.0);
                           
                            particle.logJp = beta * trace;
                        }
                        else
                        {
                            particle.logJp = 0;
                            float deltaGammaI = frobNrm + (elasticityRatio + 1.0) * trace * alpha;

                            if (deltaGammaI > 0)
                            {
                                // Project to yield surface
                                float3 h = eDiag - deltaGammaI / frobNrm * (eDiag - float3(trace / 3.0, trace / 3.0, trace / 3.0));
                                svdResult.Sigma = exp(h);
                            }
                        }
                        particle.deformationGradient = expandToFloat4x4(mul(mul(svdResult.U, diag(svdResult.Sigma)), svdResult.Vt));
                    }

                    //else if (particle.material == MaterialVisco)
                    //{
                    //    float plasticity = 0.9f;
                    //    float yieldSurface = exp(1.0 - plasticity);

                    //    // Calculate current volume
                    //    float J = svdResult.Sigma.x * svdResult.Sigma.y * svdResult.Sigma.z;  // Changed for 3D
                    //    

                    //    svdResult.Sigma = clamp(svdResult.Sigma,
                    //        float3(1.0 / yieldSurface, 1.0 / yieldSurface, 1.0 / yieldSurface),
                    //        float3(yieldSurface, yieldSurface, yieldSurface));
                    //    
                    //    float newJ = svdResult.Sigma.x * svdResult.Sigma.y * svdResult.Sigma.z;
                    //    
                    //    svdResult.Sigma *= pow(J / newJ, 1.0 / 3.0);  // Changed for 3D: using cube root
                    //    

                    //    particle.deformationGradient = mul(mul(svdResult.U, diag(svdResult.Sigma)), svdResult.Vt);
                    //}
                     
                    //else if (particle.material == MaterialSnow) {

                    //   

                    //    // Snow, snowCriticalCompression = 0.025, snowCriticalStretch = 0.0075,  snowHardeningCoeff = 10.0
                    //    //parameter and method reference based on A material point method for snow simulation
                    //    // Alexey Stomakhin, Craig Schroeder, Lawrence Chai, Joseph Teran, Andrew Selle 
                    //    //University of California Los Angeles and Walt Disney Animation Studios
                    //   //https://media.disneyanimation.com/uploads/production/publication_asset/94/asset/SSCTS13_2.pdf


                    //    // Snow parameters
                    //    float criticalCompression = 1.5e-2;
                    //    float criticalStretch = 7.5e-3;
                    //    float hardeningCoeff = 15.0;

                    //    float3 elasticSigma = clamp(svdResult.Sigma,
                    //        float3(1.0f - criticalCompression, 1.0f - criticalCompression, 1.0f - criticalCompression),
                    //        float3(1.0f + criticalStretch, 1.0f + criticalStretch, 1.0f + criticalStretch));

                    //    float Je = elasticSigma.x * elasticSigma.y * elasticSigma.z;

                    //    // Calculate hardening based on elastic volume change
                    //    float hardening = exp(hardeningCoeff * (1.0f - Je));

                    //    // Reconstruct elastic part Fe
                    //    float3x3 Fe = mul(mul(svdResult.U, diag(elasticSigma)), svdResult.Vt);

                    //    // Update plastic part Fp = F * Fe^(-1)
                    //    float3x3 FeInverse = mul(mul(svdResult.U, diag(1.0f / elasticSigma)), svdResult.Vt);
                    //    float3x3 Fp = mul(particle.deformationGradient, FeInverse);

                    //    // Update total deformation gradient F = Fe * Fp
                    //    particle.deformationGradient = expandToFloat4x4(mul(Fe * hardening, Fp));
                    //}
                   
                }
                
                // Update particle position
                particle.position += particle.displacement;

                // Mouse Iteraction Here

                /*if (g_simConstants.mouseActivation > 0)

                {
                    float3 offset = particle.position - g_simConstants.mousePosition;
                    float lenOffset = max(length(offset), 0.0001);
                    if (lenOffset < g_simConstants.mouseRadius)
                    {
                        float3 normOffset = offset / lenOffset;

                        if (g_simConstants.mouseFunction == 0)
                        {
                            particle.displacement += normOffset * 500.f;
                        }
                        else if (g_simConstants.mouseFunction == 1)
                        {
                            particle.displacement = g_simConstants.mouseVelocity * g_simConstants.deltaTime;
                        }
                    }
                }*/
                

                // Gravity Acceleration is normalized to the vertical size of the window
                particle.displacement.y -= float(g_simConstants.gridSize.y) * g_simConstants.gravityStrength * g_simConstants.deltaTime * g_simConstants.deltaTime * g_simConstants.deltaTime;

                // Free count may be negative because of emission. So make sure it is at last zero before incrementing.
                int originalMax; // Needed for InterlockedMax output parameter
                InterlockedMax(g_freeIndices[0], 0, originalMax);

                particle.position = projectInsideGuardian(particle.position, g_simConstants.gridSize, GuardianSize);
            }
            
            // Save the particle back to the buffer
            g_particles[myParticleIndex] = particle;        
        }
        
        {
            // Particle update
            if (particle.material == MaterialLiquid)
            {
                // Simple liquid viscosity: just remove deviatoric part of the deformation displacement
                float3x3 deviatoric = -1.0 * (particle.deformationDisplacement + transpose(particle.deformationDisplacement));
                particle.deformationDisplacement += expandToFloat4x4(g_simConstants.liquidViscosity * 0.5 * deviatoric);

                float alpha = 0.5 * (1.0 / particle.liquidDensity - tr(particle.deformationDisplacement) - 1.0);
                particle.deformationDisplacement += expandToFloat4x4(g_simConstants.liquidRelaxation * alpha * Identity);
            }
            else if (particle.material == MaterialSand)
            {
                // Calculate deformation gradient F
                float3x3 F = mul(Identity + particle.deformationDisplacement, particle.deformationGradient);
                SVDResult svdResult = svd(F);

                if (particle.logJp == 0)
                {
                    svdResult.Sigma = clamp(svdResult.Sigma, float3(1, 1, 1), float3(1000, 1000, 1000));
                }
                
                // Calculate closest matrix to F with det == 1
                float df = det(F);
                float cdf = clamp(abs(df), 0.1, 1.0);
                float3x3 Q = mul((1.0 / (sign(df) * sqrt(cdf))), F);

                float3x3 sigmaMat = diag(svdResult.Sigma);

                float elasticityRatio = 0.9f;
                float alpha = elasticityRatio;
                float3x3 tgt =  alpha * mul(mul(svdResult.U, sigmaMat), svdResult.Vt) + (1.0 - alpha) * Q;

                // Calculate and apply displacement difference
                float3x3 invDefGrad = inverse(particle.deformationGradient);
                float3x3 diff = mul(tgt, invDefGrad) - Identity - particle.deformationDisplacement;
                particle.deformationDisplacement += expandToFloat4x4(elasticityRatio * diff);

                // Apply viscosity
                float3x3 deviatoric = -1.0 * (particle.deformationDisplacement +
                    transpose(particle.deformationDisplacement));
                particle.deformationDisplacement += expandToFloat4x4(g_simConstants.liquidViscosity * 0.5 * deviatoric);
            }
            //else if (particle.material == MaterialElastic || particle.material == MaterialVisco) {
            //    // Calculate total deformation gradient
            //    float2x2 F = mul(Identity + particle.deformationDisplacement, particle.deformationGradient);
            //    SVDResult svdResult = svd(F);

            //    float elasticRelaxation = 1.5f;
            //    float elasticityRatio = 0.1f;

            //    // Calculate matrix closest to F with determinant = 1 (volume preserving)
            //    float df = det(F);
            //    float cdf = clamp(abs(df), 0.1, 1000.0);
            //    float2x2 Q = mul((1.0 / (sign(df) * sqrt(cdf))), F);

            //    // Interpolate between rotation (svdResult.U * svdResult.Vt) and 
            //    // volume preserving (Q) target shapes
            //    float alpha = elasticityRatio;
            //    float2x2 rotationPart = mul(svdResult.U, svdResult.Vt);
            //    float2x2 targetState = alpha * rotationPart + (1.0 - alpha) * Q;

            //    // Calculate displacement difference
            //    float2x2 invDefGrad = inverse(particle.deformationGradient);
            //    float2x2 diff = mul(targetState, invDefGrad) - Identity - particle.deformationDisplacement;

            //    // Apply relaxation
            //    particle.deformationDisplacement += elasticRelaxation * diff;
            //}
            else if (particle.material == MaterialSnow) {

                float3x3 F = mul(Identity + particle.deformationDisplacement, particle.deformationGradient);
                SVDResult svdResult = svd(F);

                // Use different parameters for displacement update to avoid overshooting
                float criticalCompression = 2.5e-2;  // Increased to allow more compression in displacement
                float criticalStretch = 5.0e-3;      // Reduced to limit stretching
                float hardeningCoeff = 10.0;         // Reduced to avoid double-hardening
                float snowViscosity = 0.1f;          // Increased for more stability
                float repulsionStrength = 0.15f;     // Adjusted for balance

                // Calculate elastic component
                float3 elasticSigma = clamp(svdResult.Sigma,
                    float3(1.0f - criticalCompression, 1.0f - criticalCompression, 1.0f - criticalCompression),
                    float3(1.0f + criticalStretch, 1.0f + criticalStretch, 1.0f + criticalStretch));

                float Je = elasticSigma.x * elasticSigma.y * elasticSigma.z;

                // Modified hardening - use smaller coefficient since we already applied hardening
                float hardening = exp(hardeningCoeff * (1.0f - Je) * 0.5);

                // Improved compression handling
                if (Je < 0.85) { // Increased threshold
                    // Use quadratic falloff for smoother response
                    float compressionRatio = (0.85 - Je) / 0.85;
                    float repulsion = repulsionStrength * compressionRatio * compressionRatio;

                    // Add directional repulsion based on SVD
                    float3x3 repulsionDir = mul(mul(svdResult.U, Identity), svdResult.Vt);
                    particle.deformationDisplacement += expandToFloat4x4(repulsion * repulsionDir);
                }

                // Update displacement with modified elastic response
                float3x3 Fe = mul(mul(svdResult.U, diag(elasticSigma)), svdResult.Vt);
                float3x3 invF = inverse(F);
                float3x3 diff = mul(Fe * hardening, invF) - Identity - particle.deformationDisplacement;

                // Apply displacement update with relaxation
                float relaxationRate = 0.8f;  // Slower update for stability
                particle.deformationDisplacement += expandToFloat4x4(snowViscosity * diff * relaxationRate);

                // Enhanced viscosity handling
                float3x3 deviatoric = -1.0 * (particle.deformationDisplacement + transpose(particle.deformationDisplacement));

                float3x3 dampingMatrix = mul(mul(svdResult.U,
                    float3x3(
                        1.2, 0, 0,    // More damping in primary compression direction
                        0, 1.0, 0,    // Normal damping in secondary direction
                        0, 0, 0.8     // Less damping in stretch direction
                    )),
                    svdResult.Vt);

                particle.deformationDisplacement += expandToFloat4x4(snowViscosity * 0.5 * mul(deviatoric, dampingMatrix));

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
                    for (int k = 0; k < 3; k++) 
                    {
                        // Weight corresponding to this neighborhood cell
                        float weight = weightInfo.weights[i].x * weightInfo.weights[j].y * weightInfo.weights[k].z;

                        // Grid vertex index
                        int3 neighborCellIndex = int3(weightInfo.cellIndex) + int3(i, j, k);

                        // 3D index relative to the corner of the local grid
                        int3 neighborCellIndexLocal = neighborCellIndex - localGridOrigin;

                        // Linear Index in the local grid
                        uint gridVertexIdx = localGridIndex(uint3(neighborCellIndexLocal));

                        // Update grid data
                        float3 offset = float3(neighborCellIndex) - p + 0.5;

                        float weightedMass = weight * particle.mass;
                        float3 momentum = weightedMass * (particle.displacement +
                            mul((float3x3)particle.deformationDisplacement, offset));
                        //float3 momentum = weightedMass * mul(particle.deformationDisplacement, offset);

                        InterlockedAdd(s_tileDataDst[gridVertexIdx + 0], encodeFixedPoint(momentum.x, g_simConstants.fixedPointMultiplier));
                        InterlockedAdd(s_tileDataDst[gridVertexIdx + 1], encodeFixedPoint(momentum.y, g_simConstants.fixedPointMultiplier));
                        InterlockedAdd(s_tileDataDst[gridVertexIdx + 2], encodeFixedPoint(momentum.z, g_simConstants.fixedPointMultiplier));
                        InterlockedAdd(s_tileDataDst[gridVertexIdx + 3], encodeFixedPoint(weightedMass, g_simConstants.fixedPointMultiplier));

                        if (g_simConstants.useGridVolumeForLiquid != 0)
                        {
                            InterlockedAdd(s_tileDataDst[gridVertexIdx + 4], encodeFixedPoint(weight * particle.volume, g_simConstants.fixedPointMultiplier));
                        }
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
        uint gridVertexAddress = gridVertexIndex(uint3(gridVertex), g_simConstants.gridSize);

        // Atomic loads from shared memory using InterlockedAdd with 0

        int dxi, dyi, dzi, wi, vi;
        InterlockedAdd(s_tileDataDst[tileDataIndex + 0], 0, dxi);
        InterlockedAdd(s_tileDataDst[tileDataIndex + 1], 0, dyi);
        InterlockedAdd(s_tileDataDst[tileDataIndex + 2], 0, dzi);
        InterlockedAdd(s_tileDataDst[tileDataIndex + 3], 0, wi);
        InterlockedAdd(s_tileDataDst[tileDataIndex + 4], 0, vi);

    // Atomic adds to the destination buffer
        InterlockedAdd(g_gridDst[gridVertexAddress + 0], dxi);
        InterlockedAdd(g_gridDst[gridVertexAddress + 1], dyi);
        InterlockedAdd(g_gridDst[gridVertexAddress + 2], dzi);
        InterlockedAdd(g_gridDst[gridVertexAddress + 3], wi);
        InterlockedAdd(g_gridDst[gridVertexAddress + 4], vi);
    
    // Clear the entries in g_gridToBeCleared
        g_gridToBeCleared[gridVertexAddress + 0] = 0;
        g_gridToBeCleared[gridVertexAddress + 1] = 0;
        g_gridToBeCleared[gridVertexAddress + 2] = 0;
        g_gridToBeCleared[gridVertexAddress + 3] = 0;
        g_gridToBeCleared[gridVertexAddress + 4] = 0;
    }

}