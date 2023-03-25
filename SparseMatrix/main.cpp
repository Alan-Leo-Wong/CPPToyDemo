/*
 * @Author: Lei Wang 602289550@qq.com
 * @Date: 2023-03-09 13:56:39
 * @LastEditors: Poet 602289550@qq.com
 * @LastEditTime: 2023-03-24 11:45:05
 * @FilePath: \ToyDemo\3slabs\main.cpp
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置
 * 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <algorithm>
#include <iostream>
#include <stdlib.h>
#include <vector>

using SpMat = Eigen::SparseMatrix<double>;
using Trip = Eigen::Triplet<double>;

template <> // function-template-specialization
struct std::less<Eigen::Vector3d> {
public:
  bool operator()(const Eigen::Vector3d &a, const Eigen::Vector3d &b) const {
    for (size_t i = 0; i < a.size(); ++i) {
      if (a[i] < b[i])
        return true;
      if (a[i] > b[i])
        return false;
    }
    return false;
  }
};

// template <>
// std::less<Eigen::VectorXd>(Eigen::VectorXd const &a, Eigen::VectorXd const
// &b) {
//   assert(a.size() == b.size());
//   for (size_t i = 0; i < a.size(); ++i) {
//     if (a[i] < b[i])
//       return true;
//     if (a[i] > b[i])
//       return false;
//   }
//   return false;
// };

int main() {
  int m = 2;
  SpMat sm(m, m);
  std::vector<Trip> matVal;

  std::map<Eigen::Vector3d, int> mp;
  mp[Eigen::Vector3d(1, 1, 1)] = 0;
  mp[Eigen::Vector3d(2, 2, 1)] = 1;
  std::cout << mp[Eigen::Vector3d(1, 1, 1)] << std::endl;

  // for (int i = 0; i < 2; ++i)
  //   for (int j = 0; j < 2; ++j)
  matVal.emplace_back(Trip(0, 0, 2));
  matVal.emplace_back(Trip(0, 0, 1)); // 此时[0,0]这个位置的值为2+1=3

  sm.setFromTriplets(matVal.begin(), matVal.end());

  // std::cout << sm.coeff(0, 1) << std::endl;

  for (const auto val : sm.coeffs())
    std::cout << val << std::endl;

  return EXIT_SUCCESS;
}
