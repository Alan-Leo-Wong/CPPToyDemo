/*
 * @Author: Poet 602289550@qq.com
 * @Date: 2023-05-12 11:22:29
 * @LastEditors: Poet 602289550@qq.com
 * @LastEditTime: 2023-05-12 13:15:18
 * @FilePath: \EigenExample\vector2matrix.cpp
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置
 * 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#include <Eigen\Dense>
#include <iostream>
#include <vector>

using namespace std;

constexpr int rows = 2;

int main() {
  std::vector<Eigen::Vector3d> vec;
  std::vector<double> vec_2;
  vec.reserve(rows);
  vec_2.reserve(rows * 3);
  for (int i = 0; i < rows; ++i) {
    vec[i] = Eigen::Vector3d(i - 1, i, i + 1);
    vec_2[i * 3] = i - 1;
    vec_2[i * 3 + 1] = i;
    vec_2[i * 3 + 2] = i + 1;
  }

  Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>> mat(
      reinterpret_cast<double *>(vec.data()), rows, 3);

  Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>> mat_2(
      reinterpret_cast<double *>(vec_2.data()), rows, 3);

  for (int i = 0; i < rows; ++i)
    std::cout << mat_2.row(i) << std::endl;
  return 0;
}
