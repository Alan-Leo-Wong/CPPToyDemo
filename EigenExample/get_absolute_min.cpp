#include <Eigen/Dense>
#include <algorithm>
#include <iostream>

int main() {
  Eigen::Array4d a, b; // 创建两个大小为4的Eigen数组

  // 初始化a和b
  a << -2.0, 5.0, -3.0, 1.0;
  b << 3.0, -4.0, 2.0, -2.0;

  // 计算绝对值最小的元素
  Eigen::Array4d result = (a.abs() < b.abs()).select(a, b);

  // 输出结果
  std::cout << "Result: " << result << std::endl;

  return 0;
}