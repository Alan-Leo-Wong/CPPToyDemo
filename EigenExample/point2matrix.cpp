#include <Eigen/Dense>
#include <iostream>
#include <vector>

int main() {
  Eigen::Vector3f f0(0, 1, 2);
  Eigen::Vector3f f1(3, 4, 5);
  std::vector<Eigen::Vector3f> vec;
  vec.emplace_back(f0);
  vec.emplace_back(f1);

  Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>> mat(
      reinterpret_cast<float *>(vec.data()), vec.size(), 3);
  for (int i = 0; i < mat.rows(); ++i)
    std::cout << mat.row(i) << std::endl;

  return 0;
}