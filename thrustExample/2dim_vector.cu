#include <driver_types.h>
#include <iostream>
#include <thrust/device_vector.h>
#include <vector>

using namespace thrust;

const int depth = 5;

__global__ void func() {}

int main() {
  std::vector<std::vector<int>> vec(depth);
  for (int i = 0; i < depth; ++i) {
    vec[i].resize(10, 0);
  }

  int **d_vec;
  cudaMalloc((void **)(&d_vec), depth * 10 * sizeof(int));

  for (int i = 0; i < depth; ++i) {
    int *d_data;
    cudaMalloc((void **)(&d_data), vec[i].size() * sizeof(int));

    cudaMemcpy(d_data, &vec[i][0], vec[i].size() * sizeof(int),
               cudaMemcpyHostToDevice);
  }

  return 0;
}