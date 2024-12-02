#include "SurfaceBlockDetectionRootSig.hlsl"
#include "../constants.h"

struct Block {
    int nonEmptyCellCount;
};

struct NumberOfBlocks {
    int value;
};

ConstantBuffer<NumberOfBlocks> numberOfBlocks : register(b0);

// SRV for blocks buffer (input buffer)
StructuredBuffer<Block> blocks : register(t0);

// UAV for the surface block indices buffer (output buffer)
RWStructuredBuffer<int> surfaceBlockIndices : register(u0);

RWStructuredBuffer<int3> surfaceBlockDispatch : register(u1);

/*
    Rather than follow the paper directly, which uses an atomic add per-thread to get a write-index into the surfaceBlockIndices buffer,
    we'll use wave intrinsics to reduce the number of atomics to one-per-wave. Parallel prefix sum might be even better, but its more complex and I wasn't able to find
    a library to do it for me.

    Inspired by this article on "stream compaction using wave intrinsics:" https://interplayoflight.wordpress.com/2022/12/25/stream-compaction-using-wave-intrinsics/
*/
[numthreads(SURFACE_BLOCK_DETECTION_THREADS_X, 1, 1)]
void main(int3 globalThreadId : SV_DispatchThreadID) {
    if (globalThreadId.x >= numberOfBlocks.value) {
        return;
    }

    Block block = blocks[globalThreadId.x];
    bool isSurfaceBlock = block.nonEmptyCellCount > 0 && block.nonEmptyCellCount < FILLED_BLOCK;
    int localSurfaceBlockIndexInWave = WavePrefixCountBits(isSurfaceBlock);
    int surfaceBlockWaveCount = WaveActiveCountBits(isSurfaceBlock);
    int surfaceBlockGlobalStartIdx;

    if (WaveIsFirstLane()) {
        InterlockedAdd(surfaceBlockDispatch[0].x, surfaceBlockWaveCount, surfaceBlockGlobalStartIdx);
        // No synchronziation is necessary, because we only care about the value within a wave.
    }
    surfaceBlockGlobalStartIdx = WaveReadLaneFirst(surfaceBlockGlobalStartIdx); // (its faster to use an intrinsic here than to write to a shared variable and sync)

    if (isSurfaceBlock) {
        int surfaceBlockGlobalIdx = surfaceBlockGlobalStartIdx + localSurfaceBlockIndexInWave;
        surfaceBlockIndices[surfaceBlockGlobalIdx] = globalThreadId.x;
    }
}