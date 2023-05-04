#include <Eigen/Dense>
#include <iostream>
#include <stdint.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/pair.h>
#include <thrust/scan.h>
#include <thrust/unique.h>

// 只有最高位即第32位为1
#define MORTON_32_FLAG 0x80000000
using thrust_edge = thrust::pair<Eigen::Vector3f, Eigen::Vector3f>;
using thrust_vert = thrust::pair<Eigen::Vector3d, uint32_t>;
using type = thrust::pair<thrust_edge, uint32_t>;
using node_vertex_type = thrust::pair<Eigen::Vector3d, uint32_t>;

struct uniqueVert {
  __host__ __device__ bool operator()(const node_vertex_type &a,
                                      const node_vertex_type &b) {
    return (a.first).isApprox(b.first, 1e-9);
  }
};

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
  // thrust::device_vector<type> d_nodeEdgeArray(3);
  // d_nodeEdgeArray[0] = type(thrust_edge(Eigen::Vector3f(0.0, 0.0, 0.0),
  //                                       Eigen::Vector3f(1.0, 1.0, 1.0)),
  //                           1);
  // d_nodeEdgeArray[1] = type(thrust_edge(Eigen::Vector3f(0.0, 0.0, 0.0),
  //                                       Eigen::Vector3f(1.0, 1.0, 1.0)),
  //                           2);
  // d_nodeEdgeArray[2] = type(thrust_edge(Eigen::Vector3f(1.0, 1.0, 1.0),
  //                                       Eigen::Vector3f(2.0, 2.0, 2.0)),
  //                           3);

  // auto newEnd = thrust::unique(d_nodeEdgeArray.begin(),
  // d_nodeEdgeArray.end(),
  //                              uniqueEdge());
  // newEnd = thrust::unique(d_nodeEdgeArray.begin(), d_nodeEdgeArray.end(),
  //                         uniqueEdge2());
  // const size_t newSize = newEnd - d_nodeEdgeArray.begin();
  // d_nodeEdgeArray.resize(newSize);
  // d_nodeEdgeArray.shrink_to_fit();
  // for (int i = 0; i < d_nodeEdgeArray.size(); ++i) {
  //   type p = d_nodeEdgeArray[i];
  //   std::cout << p.first.first.transpose() << ", " <<
  //   p.first.second.transpose()
  //             << ", " << p.second << std::endl;
  // }

  thrust::device_vector<thrust_vert> d_nodeVertArray(3);
  d_nodeVertArray[0] = thrust::make_pair(Eigen::Vector3d(0.0, 0.0, 0.0), 1);
  d_nodeVertArray[1] = thrust::make_pair(Eigen::Vector3d(0.0, 0.0, 0.0), 2);
  d_nodeVertArray[2] = thrust::make_pair(Eigen::Vector3d(1.0, 1.0, 1.0), 3);

  auto newEnd = thrust::unique(d_nodeVertArray.begin(), d_nodeVertArray.end(),
                               uniqueVert<thrust_vert>());
  const size_t newSize = newEnd - d_nodeVertArray.begin();
  d_nodeVertArray.resize(newSize);
  d_nodeVertArray.shrink_to_fit();
  for (int i = 0; i < d_nodeVertArray.size(); ++i) {
    thrust_vert p = d_nodeVertArray[i];
    std::cout << p.first.transpose() << ", " << p.second << std::endl;
  }

  // thrust::device_vector<node_vertex_type> d_nodeVertArray(3);
  // d_nodeVertArray[0] = thrust::make_pair(Eigen::Vector3d(0.0, 0.0, 0.0), 1);
  // d_nodeVertArray[1] = thrust::make_pair(Eigen::Vector3d(0.0, 0.0, 0.0), 2);
  // d_nodeVertArray[2] = thrust::make_pair(Eigen::Vector3d(1.0, 1.0, 1.0), 3);

  // auto newEnd = thrust::unique(d_nodeVertArray.begin(), d_nodeVertArray.end(),
  //                              uniqueVert());
  // const size_t newSize = newEnd - d_nodeVertArray.begin();
  // d_nodeVertArray.resize(newSize);
  // d_nodeVertArray.shrink_to_fit();
  // for (int i = 0; i < d_nodeVertArray.size(); ++i) {
  //   node_vertex_type p = d_nodeVertArray[i];
  //   std::cout << p.first.transpose() << ", " << p.second << std::endl;
  // }

  return 0;
}