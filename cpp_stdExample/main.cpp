#include <Eigen/Dense>
#include <algorithm>
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
  std::vector<Eigen::Vector3d> vec;
  vec.reserve(size);
  std::transform(nodes.begin(), nodes.end(), std::inserter(vec, vec.end()),
                 [](const Node<double> &node) {
                   return Eigen::Vector3d(node.width, node.width, node.width);
                 });
  for (const auto &val : vec)
    std::cout << val.transpose() << std::endl;
  return 0;
}