#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/scan.h>

// 只有最高位即第32位为1
#define MORTON_32_FLAG 0x80000000

template <typename T> struct sumFlag : public thrust::binary_function<T, T, T> {
  __host__ __device__ T operator()(const T &, const T &b) {
    // printf("%lu %d\n", b, (b >> 31) & 1);
    return (b >> 31) & 1;
  }
};

// unsigned int sum[256 * 256 * 256];
unsigned int arr[] = {MORTON_32_FLAG, MORTON_32_FLAG, MORTON_32_FLAG};

int main() {
  // 测试自定义求和
  thrust::host_vector<unsigned int> sum;
  sum.resize(64 * 64 * 64);
  thrust::exclusive_scan(arr, arr + 3, sum.begin(), 0, sumFlag<unsigned int>());
  for (int i = 0; i < 3; ++i)
    std::cout << sum[i] << ", ";
  std::cout << sum.size() << std::endl;

  // 测试是否可以对bool数组进行求和操作（可以）
  //   bool arr_1[4] = {0, 1, 1, 1};
  //   thrust::exclusive_scan(arr_1, arr_1 + 4, sum, 0);
  //   for (int i = 0; i < 4; ++i)
  //     std::cout << sum[i] << ", ";
  thrust::inclusive_scan(sum.begin(), sum.end(), sum.begin());
  for (int i = 0; i < 3; ++i)
    std::cout << sum[i] << ", ";
  std::cout << sum.size() << std::endl;
  return 0;
}