#include "MarchingCubes.h"

__global__ void determineKernel()
{

}

inline void preDetermineKernel(const uint &nVoxels) {
    dim3 nThreads(256, 1, 1);
    dim3 nBlocks((nVoxels + nThreads.x - 1) / nThreads.x, 1, 1);
    if (nBlocks.x > 65535) {
      nBlocks.y = nBlocks.x / 32768;
      nBlocks.x = 32768;
    }
}

void initVoxels(int argc, char **argv) {
    uint3 grid = make_uint3(20, 20, 20); // resolution
    uint nVoxels = grid.x * grid.y * grid.z;
    uint maxVerts = nVoxels * 10;
  
    float3 gridOrigin = make_float3(0, 0, 0);   // origin coordinate of grid
    float3 gridWidth = make_float3(20, 20, 20); // with of grid
    float3 voxelSize = make_float3(gridWidth.x / grid.x, gridWidth.y / grid.y,
                                   gridWidth.z / grid.z);
  
    preDetermineKernel(nVoxels);
  }
