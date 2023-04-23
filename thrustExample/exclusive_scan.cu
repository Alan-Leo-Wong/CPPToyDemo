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
  // // 测试自定义求和
  // thrust::host_vector<unsigned int> sum;
  // sum.resize(64 * 64 * 64);
  // // thrust::exclusive_scan(arr, arr + 3, sum.begin(), 0, sumFlag<unsigned
  // // int>());
  // thrust::transform(arr, arr + 3, sum.begin(), scanMortonFlag<unsigned
  // int>()); for (int i = 0; i < 3; ++i)
  //   std::cout << sum[i] << ", ";
  // std::cout << "\n--------\n";
  // thrust::exclusive_scan(sum.begin(), sum.end(), sum.begin());
  // for (int i = 0; i < 3; ++i)
  //   std::cout << sum[i] << ", ";
  // std::cout << sum.size() << std::endl;

  // // 测试是否可以对bool数组进行求和操作（可以）
  // bool arr_1[4] = {0, 1, 1, 1};
  // thrust::exclusive_scan(arr_1, arr_1 + 4, sum, 0);
  // for (int i = 0; i < 4; ++i)
  //   std::cout << sum[i] << ", ";

  // thrust::inclusive_scan(sum.begin(), sum.end(), sum.begin());
  // for (int i = 0; i < 3; ++i)
  //   std::cout << sum[i] << ", ";
  // std::cout << sum.size() << std::endl;

  // // temp
  // int arr[4] = {3, 4, 6, 7};
  // for (int i = 0; i < 4; ++i) {
  //   if (i != 3)
  //     printf("%d, ", arr[i] + 18);
  //   else
  //     printf("%d", arr[i] + 18);
  // }

  thrust::device_vector<thrust::pair<int, int>> d_nodeVertArray(3);
  d_nodeVertArray[0] = {0, 1};
  d_nodeVertArray[1] = {0, 2};
  d_nodeVertArray[2] = {1, 1};

  // for (int i = 0; i < d_nodeVertArray.size(); ++i)
  //   std::cout << ((d_nodeVertArray.data() + i)).get()->first << ", "
  //             << ((d_nodeVertArray.data() + i)).get()->second <<
  //             std::endl;
  thrust::unique(d_nodeVertArray.begin(), d_nodeVertArray.end(),
                 uniqueVert<thrust::pair<int, int>>());
  for (int i = 0; i < 3; ++i) {
    thrust::pair<int, int> p = d_nodeVertArray[i];
    std::cout << p.first << ", " << p.second << std::endl;
  }

  // std::cout << d_nodeVertArray.size() << std::endl;
  // for (int i = 0; i < d_nodeVertArray.size(); ++i)
  //   std::cout << ((d_nodeVertArray.data() + i)).get()->first << ", "
  //             << ((d_nodeVertArray.data() + i)).get()->second <<
  //             std::endl;
  return 0;
}