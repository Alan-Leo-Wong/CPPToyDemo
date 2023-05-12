/*
 * @Author: Poet 602289550@qq.com
 * @Date: 2023-05-12 11:22:29
 * @LastEditors: Poet 602289550@qq.com
 * @LastEditTime: 2023-05-12 13:26:34
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
  Eigen::Matrix<double, Eigen::Dynamic, 3> c_mat_1(rows, 3);
  for (int i = 0; i < rows; ++i)
    c_mat_1.row(i) = Eigen::Vector3d(i - 1, i, i + 1);

  Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> r_mat = c_mat_1;

  for (int i = 0; i < rows; ++i)
    std::cout << r_mat.row(i) << std::endl;

  std::cout << "=========\n";

  Eigen::MatrixXd r_mat_2 = r_mat; // 接收后成为行优先
  for (int i = 0; i < rows; ++i)
    std::cout << r_mat_2.row(i) << std::endl;

  std::cout << "=========\n";

  Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::ColMajor> c_mat_2 =
      r_mat; // 强制成为列优先
  for (int i = 0; i < rows; ++i)
    std::cout << c_mat_2.row(i) << std::endl;

  return 0;
}
