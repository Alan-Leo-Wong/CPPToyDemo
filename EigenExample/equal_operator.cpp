/*
 * @Author: WangLei
 * @authorEmail: leiw1006@gmail.com
 * @Date: 2023-12-21 11:01:48
 * @LastEditors: WangLei
 * @LastEditTime: 2023-12-21 11:07:02
 * @FilePath: \EigenExample\equal_operator.cpp
 * @Description:
 */
#include <Eigen/Dense>
#include <iostream>

int main() {
  // 定义四维向量
  Eigen::Vector4d a(1, 2, 3, 3);

  // 定义偏移值
  double abs_offsetdis = 3;

  // 找到等于abs_offsetdis的元素的索引
  Eigen::Vector4d indices = (a.array() == abs_offsetdis).cast<double>();
  std::cout << "Indices: " << (indices * 1e-6).transpose() << std::endl;

  // 添加扰动项
  a += (indices * 1e-1);

  // 打印结果
  std::cout << "Updated vector: " << a.transpose() << std::endl;

  return 0;
}
