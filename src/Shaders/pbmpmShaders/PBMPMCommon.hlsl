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

#define ShapeTypeBox 0
#define ShapeTypeCircle 1

#define ShapeFunctionEmit 0
#define ShapeFunctionCollider  1
#define ShapeFunctionDrain  2
#define ShapeFunctionInitialEmit  3

#define TotalBukkitEdgeLength (BukkitSize + BukkitHaloSize * 2)
#define TileDataSizePerEdge (TotalBukkitEdgeLength * 3)
#define VolumeTileDataSize (TotalBukkitEdgeLength * 2 * TotalBukkitEdgeLength * 2)
#define TileDataSize (TileDataSizePerEdge * TileDataSizePerEdge)

struct PBMPMConstants {
	uint2 gridSize;
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
	float2 position;
	float2 displacement;
	float2x2 deformationGradient;
	float2x2 deformationDisplacement;

	float liquidDensity;
	float mass;
	float material;
	float volume;

	float lambda;
	float logJp;
	float enabled;
};

struct BukkitThreadData {
	unsigned int rangeStart;
	unsigned int rangeCount;
	unsigned int bukkitX;
	unsigned int bukkitY;
};

struct SimShape {
	int id;
	float2 position;
	float rotation;
	float2 halfSize;

	int shapeType;
	int functionality;
	int material;
	float emissionRate;
	int radius;
};

// Helper Functions

// Function to calculate the grid vertex index using lexicographical ordering
uint gridVertexIndex(uint2 gridVertex, uint2 gridSize)
{
	// 4 components per grid vertex
	return 4 * (gridVertex.y * gridSize.x + gridVertex.x);
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
    float2 weights[3];
    float2 cellIndex;
};

// Helper function for element-wise square (power of 2)
float2 pow2(float2 x)
{
    return x * x;
}

// Initialize QuadraticWeightInfo based on position
QuadraticWeightInfo quadraticWeightInit(float2 position)
{
    float2 roundDownPosition = floor(position);
    float2 offset = (position - roundDownPosition) - 0.5;

    QuadraticWeightInfo result;
    result.weights[0] = 0.5 * pow2(0.5 - offset);
    result.weights[1] = 0.75 - pow2(offset);
    result.weights[2] = 0.5 * pow2(0.5 + offset);
    result.cellIndex = roundDownPosition - float2(1, 1);

    return result;
}

// Helper function for element-wise cube (power of 3)
float2 pow3(float2 x)
{
    return x * x * x;
}

// Structure to hold cubic weight information
struct CubicWeightInfo
{
    float2 weights[4];
    float2 cellIndex;
};

// Initialize CubicWeightInfo based on position
CubicWeightInfo cubicWeightInit(float2 position)
{
    float2 roundDownPosition = floor(position);
    float2 offset = position - roundDownPosition;

    CubicWeightInfo result;
    result.weights[0] = pow3(2.0 - (1.0 + offset)) / 6.0;
    result.weights[1] = 0.5 * pow3(offset) - pow2(offset) + float2(2.0 / 3.0, 2.0 / 3.0);
    result.weights[2] = 0.5 * pow3(1.0 - offset) - pow2(1.0 - offset) + float2(2.0 / 3.0, 2.0 / 3.0);
    result.weights[3] = pow3(2.0 - (2.0 - offset)) / 6.0;
    result.cellIndex = roundDownPosition - float2(1, 1);

    return result;
}

// Bukkit and Dispatch helpers 

uint bukkitAddressToIndex(uint2 bukkitAddress, uint bukkitCountX)
{
    return bukkitAddress.y * bukkitCountX + bukkitAddress.x;
}

int2 positionToBukkitId(float2 position)
{
    return int2(position / float(BukkitSize));
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
