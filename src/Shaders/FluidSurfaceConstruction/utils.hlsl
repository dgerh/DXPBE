inline int to1D(int3 index, int3 dimensions) {
    return index.x + (index.y * dimensions.x) + (index.z * dimensions.x * dimensions.y);
}

// Use sparingly, prefer having things in 1D if possible. (Modulus and division are expensive)
int3 to3D(int index, int3 dimensions) {
    int x = index % dimensions.x;
    int y = (index / dimensions.x) % dimensions.y;
    int z = index / (dimensions.x * dimensions.y);
    return int3(x, y, z);
}

inline float cubic(float x) {
    return x * x * x;
}

inline int divRoundUp(int num, int denom)
{
    return (num + denom - 1) / denom;
}