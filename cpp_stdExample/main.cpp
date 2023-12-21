/*
 * @Author: WangLei
 * @authorEmail: leiw1006@gmail.com
 * @Date: 2023-05-04 09:20:51
 * @LastEditors: WangLei
 * @LastEditTime: 2023-05-05 10:46:48
 * @FilePath: \cpp_stdExample\main.cpp
 * @Description:
 */
// #include <Eigen/Dense>
#include "test.h"
#include <algorithm>
#include <functional>
#include <iostream>
#include <iterator>
#include <vector>

template <typename T> struct Node {
  T width;
  Node() : width(T(3.0)) {}
};

constexpr int size = 10;
std::vector<Node<double>> nodes(size);

int main() {
  // std::vector<Eigen::Vector3d> vec;
  // vec.reserve(size);
  // std::transform(nodes.begin(), nodes.end(), std::inserter(vec, vec.end()),
  //                [](const Node<double> &node) {
  //                  return Eigen::Vector3d(node.width, node.width,
  //                  node.width);
  //                });
  // for (const auto &val : vec)
  //   std::cout << val.transpose() << std::endl;

  const auto time = [](std::function<void(void)> func) -> double {
    const double t_before = igl::get_seconds();
    func();
    const double t_after = igl::get_seconds();
    return t_after - t_before;
  };

  return 0;
}