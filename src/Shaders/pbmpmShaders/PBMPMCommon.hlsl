// Keep consistent with PBMPMScene.h

#define ParticleDispatchSize 64
#define GridDispatchSize 8
#define BukkitSize 6
#define BukkitHaloSize 1
#define GuardianSize 3

#define MaterialLiquid 0
#define MaterialElastic 1
#define MaterialSand 2
#define MaterialVisco 3
#define MaterialSnow 4

#define ShapeTypeBox 0
#define ShapeTypeCircle 1

#define ShapeFunctionEmit 0
#define ShapeFunctionCollider  1
#define ShapeFunctionDrain  2
#define ShapeFunctionInitialEmit  3

#define TotalBukkitEdgeLength (BukkitSize + BukkitHaloSize * 2)
#define TileDataSizePerEdge (TotalBukkitEdgeLength * 5) //4->5
#define TileDataSize (TileDataSizePerEdge * TileDataSizePerEdge)

struct PBMPMConstants {
	uint3 gridSize; //2 -> 3
	float deltaTime;
	float gravityStrength;

	float liquidRelaxation;
	float liquidViscosity;
	unsigned int fixedPointMultiplier;

	unsigned int useGridVolumeForLiquid;
	unsigned int particlesPerCellAxis;

	float frictionAngle;
	unsigned int shapeCount;
	unsigned int simFrame;

	unsigned int bukkitCount;
	unsigned int bukkitCountX;
	unsigned int bukkitCountY;
	unsigned int bukkitCountZ; //added Z
	unsigned int iteration;
	unsigned int iterationCount;
	float borderFriction;


    //mouse stuff
    unsigned int mouseActivation;
    uint2 mousePosition;
    unsigned int mouseRadius;
    unsigned int mouseFunction;
    unsigned int mouseVelocity;

};


// Define constants for identity and zero matrices
static const float2x2 Identity = float2x2(1, 0, 0, 1);
static const float2x2 ZeroMatrix = float2x2(0, 0, 0, 0);

struct Particle {
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

struct BukkitThreadData {
	unsigned int rangeStart;
	unsigned int rangeCount;
	unsigned int bukkitX;
	unsigned int bukkitY;
	unsigned int bukkitZ; //added Z
};

struct SimShape {
    int id;
    float3 position;
    float rotation;
    float3 halfSize;

    int shapeType;
    int functionality;
    int material;
    float emissionRate;
    int radius;
};

// Helper Functions

// Function to calculate the grid vertex index using lexicographical ordering
uint gridVertexIndex(uint3 gridVertex, uint3 gridSize)
{
	// 5 components per grid vertex -- xyz and 2 weights
	return 5 * (gridVertex.z * gridVertex.y * gridVertex.x + gridVertex.y * gridSize.x + gridVertex.x);
}

// Function to decode a fixed-point integer to a floating-point value
float decodeFixedPoint(int fixedPoint, uint fixedPointMultiplier)
{
	return float(fixedPoint) / float(fixedPointMultiplier);
}

// Function to encode a floating-point value as a fixed-point integer
int encodeFixedPoint(float floatingPoint, uint fixedPointMultiplier)
{
	return int(floatingPoint * float(fixedPointMultiplier));
}

// Structure to hold quadratic weight information
struct QuadraticWeightInfo
{
    float3 weights[3]; //not rly sure what this is for... these used to be float2s
    float3 cellIndex;
};

// Helper function for element-wise square (power of 2)
float3 pow2(float3 x) {
	return x * x;
}

// Initialize QuadraticWeightInfo based on position
QuadraticWeightInfo quadraticWeightInit(float3 position)
{
    float3 roundDownPosition = floor(position);
    float3 offset = (position - roundDownPosition) - 0.5;

    QuadraticWeightInfo result;
    result.weights[0] = 0.5 * pow2(0.5 - offset);
    result.weights[1] = 0.75 - pow2(offset);
    result.weights[2] = 0.5 * pow2(0.5 + offset);
    result.cellIndex = roundDownPosition - float3(1, 1, 1);

    return result;
}

// Helper function for element-wise cube (power of 3)
float3 pow3(float3 x) {
	return x * x * x;
}

// Structure to hold cubic weight information
struct CubicWeightInfo
{
    float3 weights[4];
    float3 cellIndex;
};

// Initialize CubicWeightInfo based on position
CubicWeightInfo cubicWeightInit(float3 position)
{
    float3 roundDownPosition = floor(position);
    float3 offset = position - roundDownPosition;

    CubicWeightInfo result;
    result.weights[0] = pow3(2.0 - (1.0 + offset)) / 6.0;
    result.weights[1] = 0.5 * pow3(offset) - pow2(offset) + float3(2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0); // just add a 2/3, may need to adjust
    result.weights[2] = 0.5 * pow3(1.0 - offset) - pow2(1.0 - offset) + float3(2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0);
    result.weights[3] = pow3(2.0 - (2.0 - offset)) / 6.0;
    result.cellIndex = roundDownPosition - float3(1, 1, 1);

    return result;
}

// Bukkit and Dispatch helpers 

uint bukkitAddressToIndex(uint3 bukkitAddress, uint bukkitCountX, uint bukkitCountY)
{
    return bukkitAddress.z * bukkitCountY * bukkitCountX + bukkitAddress.y * bukkitCountX + bukkitAddress.x;
}

int3 positionToBukkitId(float3 position)
{
    return int3(position / float(BukkitSize));
}

uint divUp(uint threadCount, uint groupSize)
{
    return (threadCount + groupSize - 1) / groupSize;
}

// Collision

float2x2 rot(float angle)
{
    float c = cos(angle);
    float s = sin(angle);
    return float2x2(c, -s, s, c);
}



struct CollideResult
{
    bool collides;
    float penetration;
    float2 normal;
    float2 pointOnCollider;
};

CollideResult collide(SimShape shape, float2 pos)
{
    CollideResult result;
    if (shape.shapeType == ShapeTypeCircle)
    {
        float2 offset = shape.position - pos;
        float offsetLen = length(offset);
        float2 normal = offset * (offsetLen == 0 ? 0 : 1.0 / offsetLen);
        result.collides = offsetLen <= shape.radius;
        result.penetration = -(offsetLen - shape.radius);
        result.normal = normal;
        result.pointOnCollider = shape.position + normal * shape.radius;
    }
    else if (shape.shapeType == ShapeTypeBox)
    {
        float2 offset = pos - shape.position;
        float2x2 R = rot(shape.rotation / 180.0f * 3.14159f); // Assuming `rot` is a 2D rotation matrix function
        float2 rotOffset = mul(R, offset); // Matrix-vector multiplication
        float sx = sign(rotOffset.x);
        float sy = sign(rotOffset.y);
        float2 penetration = -(abs(rotOffset) - shape.halfSize);
        float2 normal = mul(transpose(R), 
            (penetration.y < penetration.x ? float2(sx, 0) : float2(0, sy)));

        float minPen = min(penetration.x, penetration.y);

        float2 pointOnBox = shape.position + mul(transpose(R), clamp(rotOffset, -shape.halfSize, shape.halfSize));

        result.collides = minPen > 0;
        result.penetration = minPen;
        result.normal = -normal;
        result.pointOnCollider = pointOnBox;
    }
    else
    {
        result.collides = false;
        result.penetration = 0.0;
        result.normal = float2(0, 0);
        result.pointOnCollider = float2(0, 0);
    }

    return result;
}
