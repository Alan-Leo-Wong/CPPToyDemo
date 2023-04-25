#include <iostream>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <thrust/unique.h>

// 只有最高位即第32位为1
#define MORTON_32_FLAG 0x80000000

template <typename T>
struct scanMortonFlag : public thrust::unary_function<T, T> {
  __host__ __device__ T operator()(const T &x) {
    // printf("%lu %d\n", b, (b >> 31) & 1);
    return (x >> 31) & 1;
  }
};

template <typename T>
struct uniqueVert : public thrust::binary_function<T, T, T> {
  __host__ __device__ bool operator()(const T &a, const T &b) {
    return a.first == b.first;
  }
};

// unsigned int sum[256 * 256 * 256];
unsigned int arr[] = {MORTON_32_FLAG, MORTON_32_FLAG, MORTON_32_FLAG};

int main() {
  thrust::device_vector<thrust::pair<int, int>> d_nodeVertArray(3);
  d_nodeVertArray[0] = {0, 1};
  d_nodeVertArray[1] = {0, 2};
  d_nodeVertArray[2] = {1, 1};

  auto newEnd = thrust::unique(d_nodeVertArray.begin(), d_nodeVertArray.end(),
                               uniqueVert<thrust::pair<int, int>>());
  const size_t newSize = newEnd - d_nodeVertArray.begin();
  d_nodeVertArray.resize(newSize);
  d_nodeVertArray.shrink_to_fit();
  for (int i = 0; i < d_nodeVertArray.size(); ++i) {
    thrust::pair<int, int> p = d_nodeVertArray[i];
    std::cout << p.first << ", " << p.second << std::endl;
  }
  return 0;
}