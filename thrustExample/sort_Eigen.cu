#include <Eigen/Dense>
#include <algorithm>
#include <driver_types.h>
#include <iostream>
#include <iterator>
#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/pair.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/unique.h>

using node_vertex_type = thrust::pair<Eigen::Vector3d, uint32_t>;
using thrust_edge_type = thrust::pair<Eigen::Vector3d, Eigen::Vector3d>;
using node_edge_type = thrust::pair<thrust_edge_type, uint32_t>;

// struct sortVert {
//   __host__ __device__ bool operator()(const node_vertex_type &a,
//                                       const node_vertex_type &b) {
//     for (size_t i = 0; i < 3; ++i) {
//       if (fabs((a.first)[i] - (b.first)[i]) < 1e-9)
//         continue;
//       if ((a.first)[i] < (b.first)[i])
//         return true;
//       else if ((a.first)[i] > (b.first)[i])
//         return false;
//     }
//     return a.second < b.second;
//   }
// };

template <typename T> struct lessPoint {
  __host__ __device__ int operator()(const T &a, const T &b) const {
    for (size_t i = 0; i < a.size(); ++i) {
      if (fabs(a[i] - b[i]) < 1e-9)
        continue;

      if (a[i] < b[i])
        return 1;
      else if (a[i] > b[i])
        return -1;
    }
    return 0;
  }
};

struct sortVert {
  __host__ __device__ bool operator()(const node_vertex_type &a,
                                      const node_vertex_type &b) {
    int t = lessPoint<Eigen::Vector3d>{}(a.first, b.first);
    if (t == 0)
      return a.second < b.second;
    else if (t == 1)
      return true;
    else
      return false;
  }
};

struct transformEdge {
  __host__ __device__ node_edge_type operator()(const node_edge_type &a) {
    if (lessPoint<Eigen::Vector3d>{}(a.first.first, a.first.second) == -1) {
      node_edge_type _p;
      //   printf("a.first.second = (%lf, %lf, %lf)\n", a.first.second.x(),
      //          a.first.second.y(), a.first.second.z());
      _p.first.first = a.first.second, _p.first.second = a.first.first;
      _p.second = a.second;
      return _p;
    } else
      return a;
  }
};

__host__ __device__ int parallelAxis(const node_edge_type &e) {
  Eigen::Vector3d dir = e.first.second - e.first.first;
  if (dir.cross(Eigen::Vector3d(1, 0, 0))
          .isApprox(Eigen::Vector3d(0, 0, 0), 1e-6)) // 与x轴平行
    return 1;
  else if (dir.cross(Eigen::Vector3d(0, 1, 0))
               .isApprox(Eigen::Vector3d(0, 0, 0), 1e-6)) // 与y轴平行
    return 2;
  else if (dir.cross(Eigen::Vector3d(0, 0, 1))
               .isApprox(Eigen::Vector3d(0, 0, 0), 1e-6)) // 与z轴平行
    return 3;
}

struct sortEdge {
  __host__ __device__ bool operator()(node_edge_type &a, node_edge_type &b) {
    int _p0_case = parallelAxis(a);
    int _p1_case = parallelAxis(b);
    if (_p0_case != _p1_case)
      return _p0_case < _p1_case;

    int _t_0 = lessPoint<Eigen::Vector3d>{}(a.first.second, b.first.second);
    if (_t_0 == 0) {
      int _t_1 = lessPoint<Eigen::Vector3d>{}(a.first.first, b.first.first);
      if (_t_1 == 0)
        return a.second < b.second;
      else if (_t_1 == 1)
        return true;
      else
        return false;
    } else if (_t_0 == 1)
      return true;
    else
      return false;
  }
};

struct uniqueVert {
  __host__ __device__ bool operator()(const node_vertex_type &a,
                                      const node_vertex_type &b) {
    return (a.first).isApprox(b.first, 1e-9);
  }
};

struct lessYVal {
  bool operator()(const node_edge_type &a,
                  const node_edge_type &b) { // Search for first element 'a' in
                                             // list such that b ≤ a
    std::cout << "b.first.first = " << b.first.first.transpose()
              << ", b.first.second = " << b.first.second.transpose()
              << ", a.first.first = " << a.first.first.transpose() << std::endl;
    // return b.first.first.y() > a.first.second.y();
    return b.first.second.y() > a.first.second.y();
    // return parallelAxis(a) < parallelAxis(b);
  }
};

int main() {
  thrust::device_vector<node_edge_type> d_nodeEdgeArray(5);
  d_nodeEdgeArray[0] =
      node_edge_type(thrust_edge_type(Eigen::Vector3d(0.0, 0.0, 0.0),
                                      Eigen::Vector3d(2.0, 0.0, 0.0)),
                     1);
  d_nodeEdgeArray[1] =
      node_edge_type(thrust_edge_type(Eigen::Vector3d(0.0, 0.0, 0.0),
                                      Eigen::Vector3d(0.0, 2.0, 0.0)),
                     0);
  d_nodeEdgeArray[2] =
      node_edge_type(thrust_edge_type(Eigen::Vector3d(0.0, 0.0, 0.0),
                                      Eigen::Vector3d(0.0, 0.0, 2.0)),
                     3);

  d_nodeEdgeArray[3] =
      node_edge_type(thrust_edge_type(Eigen::Vector3d(0.0, 2.0, 2.0),
                                      Eigen::Vector3d(0.0, 3.0, 2.0)),
                     3);

  d_nodeEdgeArray[4] =
      node_edge_type(thrust_edge_type(Eigen::Vector3d(0.0, 1.0, 0.0),
                                      Eigen::Vector3d(0.0, 1.5, 0.0)),
                     3);

  thrust::device_vector<node_edge_type> td_nodeEdgeArray(5);
  thrust::transform(d_nodeEdgeArray.begin(), d_nodeEdgeArray.end(),
                    td_nodeEdgeArray.begin(), transformEdge());
  for (int i = 0; i < td_nodeEdgeArray.size(); ++i) {
    node_edge_type p = td_nodeEdgeArray[i];
    std::cout << p.first.first.transpose() << ", " << p.first.second.transpose()
              << std::endl;
  }

  std::cout << "=========\n";

  std::vector<node_edge_type> h_nodeEdgeArray(5);
  cudaMemcpy(h_nodeEdgeArray.data(), td_nodeEdgeArray.data().get(),
             sizeof(node_edge_type) * 5, cudaMemcpyDeviceToHost);
  std::sort(h_nodeEdgeArray.begin(), h_nodeEdgeArray.end(), sortEdge());

  std::vector<node_edge_type> x_nodeEdgeArray;
  std::vector<node_edge_type> y_nodeEdgeArray;
  std::vector<node_edge_type> z_nodeEdgeArray;

#pragma omp parallel
  {
    std::copy_if(
        h_nodeEdgeArray.begin(), h_nodeEdgeArray.end(),
        std::back_inserter(x_nodeEdgeArray),
        [](const node_edge_type &val) { return parallelAxis(val) == 1; });
    std::sort(x_nodeEdgeArray.begin(), x_nodeEdgeArray.end(), sortEdge());

    std::copy_if(
        h_nodeEdgeArray.begin(), h_nodeEdgeArray.end(),
        std::back_inserter(y_nodeEdgeArray),
        [](const node_edge_type &val) { return parallelAxis(val) == 2; });
    std::sort(y_nodeEdgeArray.begin(), y_nodeEdgeArray.end(), sortEdge());

    std::copy_if(
        h_nodeEdgeArray.begin(), h_nodeEdgeArray.end(),
        std::back_inserter(z_nodeEdgeArray),
        [](const node_edge_type &val) { return parallelAxis(val) == 3; });
    std::sort(z_nodeEdgeArray.begin(), z_nodeEdgeArray.end(), sortEdge());
  }

  for (int i = 0; i < x_nodeEdgeArray.size(); ++i) {
    node_edge_type p = x_nodeEdgeArray[i];
    std::cout << p.first.first.transpose() << ", " << p.first.second.transpose()
              << std::endl;
  }
  std::cout << "=========\n";

  for (int i = 0; i < y_nodeEdgeArray.size(); ++i) {
    node_edge_type p = y_nodeEdgeArray[i];
    std::cout << p.first.first.transpose() << ", " << p.first.second.transpose()
              << std::endl;
  }
  std::cout << "=========\n";

  for (int i = 0; i < z_nodeEdgeArray.size(); ++i) {
    node_edge_type p = z_nodeEdgeArray[i];
    std::cout << p.first.first.transpose() << ", " << p.first.second.transpose()
              << std::endl;
  }
  std::cout << "=========\n";

  // thrust::sort(td_nodeEdgeArray.begin(), td_nodeEdgeArray.end(),
  // sortEdge()); for (int i = 0; i < td_nodeEdgeArray.size(); ++i) {
  //   node_edge_type p = td_nodeEdgeArray[i];
  //   std::cout << p.first.first.transpose() << ", " <<
  //   p.first.second.transpose()
  //             << std::endl;
  // }
  // std::cout << "=========\n";

  // // std::vector<node_edge_type> h_nodeEdgeArray(5);
  // // cudaMemcpy(h_nodeEdgeArray.data(), td_nodeEdgeArray.data().get(),
  // //            sizeof(node_edge_type) * 5, cudaMemcpyDeviceToHost);
  // thrust::host_vector<node_edge_type> h_nodeEdgeArray = td_nodeEdgeArray;

  node_edge_type a;
  a.first.first = Eigen::Vector3d(0, 0, 0);
  a.first.second = Eigen::Vector3d(0, 2, 0);
  a.second = 0;
  auto upper = std::lower_bound(
      y_nodeEdgeArray.begin(), y_nodeEdgeArray.end(), a,
      lessYVal()); // Search for first element x such that a ≤ y
  if (upper != y_nodeEdgeArray.end()) {
    std::cout << (*upper).first.first.transpose() << ", "
              << (*upper).first.second.transpose() << std::endl;
  }
  std::cout << "=========\n";

  // h_nodeEdgeArray.erase(upper);

  // for (int i = 0; i < h_nodeEdgeArray.size(); ++i) {
  //   node_edge_type p = h_nodeEdgeArray[i];
  //   std::cout << p.first.first.transpose() << ", " <<
  //   p.first.second.transpose()
  //             << std::endl;
  // }
  // std::cout << "=========\n";

  // auto newEnd = thrust::unique(d_nodeEdgeArray.begin(),
  // d_nodeEdgeArray.end(),
  //                              uniqueEdge());
  // newEnd = thrust::unique(d_nodeEdgeArray.begin(), d_nodeEdgeArray.end(),
  //                         uniqueEdge2());
  // const size_t newSize = newEnd - d_nodeEdgeArray.begin();
  // d_nodeEdgeArray.resize(newSize);
  // d_nodeEdgeArray.shrink_to_fit();
  // for (int i = 0; i < d_nodeEdgeArray.size(); ++i) {
  //   node_edge_type p = d_nodeEdgeArray[i];
  //   std::cout << p.first.first.transpose() << ", " <<
  //   p.first.second.transpose()
  //             << ", " << p.second << std::endl;
  // }

  // thrust::device_vector<node_vertex_type> d_nodeVertArray(3);
  // d_nodeVertArray[0] = thrust::make_pair(Eigen::Vector3d(0.0, 0.0, 0.0),
  // 3); d_nodeVertArray[1] = thrust::make_pair(Eigen::Vector3d(0.0, 1.0,
  // 0.0), 3); d_nodeVertArray[2] = thrust::make_pair(Eigen::Vector3d(0.0,
  // 0.0, 1.0), 2);

  // thrust::sort(d_nodeVertArray.begin(), d_nodeVertArray.end(),
  // sortVert()); for (int i = 0; i < d_nodeVertArray.size(); ++i) {
  //   node_vertex_type p = d_nodeVertArray[i];
  //   std::cout << p.first.transpose() << ", " << p.second << std::endl;
  // }
  // std::cout << "=========\n";
  // auto newEnd = thrust::unique(d_nodeVertArray.begin(),
  // d_nodeVertArray.end(),
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