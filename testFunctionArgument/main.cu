/*
 * @Author: Poet 602289550@qq.com
 * @Date: 2023-04-29 13:57:22
 * @LastEditors: Poet 602289550@qq.com
 * @LastEditTime: 2023-05-03 11:36:30
 * @FilePath: \testFunctionArgument\main.cpp
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置
 * 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#include <algorithm>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <functional>
#include <iostream>
#include <numeric>

// static inline __host__ void getOccupancyMaxPotentialBlockSize(
//     const uint32_t &dataSize, int &minGridSize, int &blockSize, int
//     &gridSize, size_t dynamicSMemSize = 0, int blockSizeLimit = 0) {
//   cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, func,
//                                      dynamicSMemSize, blockSizeLimit);
//   gridSize = (dataSize + blockSize - 1) / blockSize;
// }

__global__ void test1() {}

__global__ void test2(const int n) {}

int main() {
  uint32_t dataSize = 32;
  int minGridSize;
  int blockSize;
  int gridSize;

  // getOccupancyMaxPotentialBlockSize(dataSize, minGridSize, blockSize,
  // gridSize,
  //                                   test);

  std::vector<uint32_t> nodeIdx(10);
  std::iota(nodeIdx.begin(), nodeIdx.end(), 0);
  for (auto val : nodeIdx)
    std::cout << val << std::endl;

  return 0;
}