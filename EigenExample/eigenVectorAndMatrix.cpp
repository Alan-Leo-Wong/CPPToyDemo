/*
 * @Author: Poet 602289550@qq.com
 * @Date: 2023-11-12 13:12:49
 * @LastEditors: Poet 602289550@qq.com
 * @LastEditTime: 2023-11-12 13:24:35
 * @FilePath: \EigenExample\eigenVectorToMatrix.cpp
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置
 * 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#include <Eigen\Dense>
#include <iostream>
#include <vector>

using namespace std;

constexpr int rows = 5;

// 实现一个n*3的向量到(n, 3)的矩阵的映射
int main() {
  Eigen::VectorXd vec(rows * 3);

  for (int i = 0; i < rows; ++i) {
    vec(i * 3) = i * 3;
    vec(i * 3 + 1) = i * 3 + 1;
    vec(i * 3 + 2) = i * 3 + 2;
  }

  // 必须使用Eigen::RowMajor，否则变换后的数据将按列优先存储，导致排布顺序错误
  Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> mat =
      Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>>(
          vec.data(), rows, 3);

  for (int i = 0; i < rows; ++i)
    std::cout << mat.row(i) << std::endl;

  // 测试再转换回去的结果
  std::cout << "==========\n";
  Eigen::VectorXd vec_2 = Eigen::Map<Eigen::VectorXd>(mat.data(), rows * 3);
  std::cout << vec_2 << std::endl;

  std::cout << "==========\n";
  Eigen::MatrixXd mat_3 = mat;
  mat_3.transposeInPlace(); // 对列优先必须加transpose
  Eigen::VectorXd vec_3 = Eigen::Map<Eigen::VectorXd>(mat_3.data(), rows * 3);
  std::cout << vec_3 << std::endl;

  return 0;
}
