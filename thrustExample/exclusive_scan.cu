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
  // 测试自定义求和
  thrust::host_vector<bool> validArray;
  thrust::host_vector<unsigned int> sum;
  validArray.resize(64 * 64 * 64);
  sum.resize(64 * 64 * 64);
  // thrust::exclusive_scan(arr, arr + 3, sum.begin(), 0, sumFlag<unsigned
  // int>());
  thrust::transform(arr, arr + 3, validArray.begin(),
                    scanMortonFlag<unsigned int>());
  for (int i = 0; i < 3; ++i)
    std::cout << validArray[i] << ", ";
  std::cout << "\n--------\n";
  // 必须加init，否则并不会得到相加的结果
  thrust::exclusive_scan(validArray.begin(), validArray.end(), sum.begin(), 0);
  for (int i = 0; i < 3; ++i)
    std::cout << sum[i] << ", ";
  std::cout << sum.size() << std::endl;

  // // 测试是否可以对bool数组进行求和操作（可以）
  // bool arr_1[4] = {0, 1, 1, 1};
  // thrust::exclusive_scan(arr_1, arr_1 + 4, sum.begin(), 0);
  // for (int i = 0; i < 4; ++i)
  //   std::cout << sum[i] << ", ";

  // thrust::inclusive_scan(sum.begin(), sum.end(), sum.begin());
  // for (int i = 0; i < 3; ++i)
  //   std::cout << sum[i] << ", ";
  // std::cout << sum.size() << std::endl;

  return 0;
}