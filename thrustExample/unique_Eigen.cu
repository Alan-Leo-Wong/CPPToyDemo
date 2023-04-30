#include <Eigen/Dense>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/pair.h>
#include <thrust/scan.h>
#include <thrust/unique.h>

// 只有最高位即第32位为1
#define MORTON_32_FLAG 0x80000000
using thrust_edge = thrust::pair<Eigen::Vector3f, Eigen::Vector3f>;
using type = thrust::pair<thrust_edge, uint32_t>;

struct uniqueEdge {
  __host__ __device__ bool operator()(const type &a, const type &b) {
    return (a.first.first == b.first.first) &&
           (a.first.second == b.first.second);
  }
};

struct uniqueEdge2 {
  __host__ __device__ bool operator()(const type &a, const type &b) {
    return (a.first.first != b.first.first) &&
           (a.first.second != b.first.second);
  }
};

int main() {
  thrust::device_vector<type> d_nodeEdgeArray(3);
  d_nodeEdgeArray[0] = type(thrust_edge(Eigen::Vector3f(0.0, 0.0, 0.0),
                                        Eigen::Vector3f(1.0, 1.0, 1.0)),
                            1);
  d_nodeEdgeArray[1] = type(thrust_edge(Eigen::Vector3f(0.0, 0.0, 0.0),
                                        Eigen::Vector3f(1.0, 1.0, 1.0)),
                            2);
  d_nodeEdgeArray[2] = type(thrust_edge(Eigen::Vector3f(1.0, 1.0, 1.0),
                                        Eigen::Vector3f(2.0, 2.0, 2.0)),
                            3);

  auto newEnd = thrust::unique(d_nodeEdgeArray.begin(), d_nodeEdgeArray.end(),
                               uniqueEdge());
  newEnd = thrust::unique(d_nodeEdgeArray.begin(), d_nodeEdgeArray.end(),
                          uniqueEdge2());
  const size_t newSize = newEnd - d_nodeEdgeArray.begin();
  d_nodeEdgeArray.resize(newSize);
  d_nodeEdgeArray.shrink_to_fit();
  for (int i = 0; i < d_nodeEdgeArray.size(); ++i) {
    type p = d_nodeEdgeArray[i];
    std::cout << p.first.first.transpose() << ", " << p.first.second.transpose()
              << ", " << p.second << std::endl;
  }
  return 0;
}