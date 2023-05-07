#include "eigen.hpp"
// #include <Eigen\Dense>
#include <iostream>

int main() {
  Eigen::Vector3d a(0, 0, 0);
  std::cout << a.transpose() << std::endl;
  return 0;
}