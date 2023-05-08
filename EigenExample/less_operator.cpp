/*
 * @Author: Poet 602289550@qq.com
 * @Date: 2023-05-07 17:00:25
 * @LastEditors: Poet 602289550@qq.com
 * @LastEditTime: 2023-05-08 19:44:22
 * @FilePath: \EigenExample\less_operator.cpp
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置
 * 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#include <Eigen\Dense>
// #include <Eigen\Dense>
#include <functional>
#include <iostream>

template <typename Scalar> struct VectorCompare {
  bool operator()(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &a,
                  const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &b) {
    for (size_t i = 0; i < a.size(); ++i) {
      if (fabs(a[i] - b[i]) < 1e-9)
        continue;

      if (a[i] < b[i])
        return true;
      else if (a[i] > b[i])
        return false;
      ;
    }
    return false;
  }
};

template <> struct std::less<Eigen::Vector3d> {
public:
  bool operator()(const Eigen::Vector3d &a, const Eigen::Vector3d &b) const {
    for (size_t i = 0; i < a.size(); ++i) {
      if (fabs(a[i] - b[i]) < 1e-9)
        continue;

      if (a[i] < b[i])
        return true;
      else if (a[i] > b[i])
        return false;
    }
    return false;
  }
};

template <typename A, typename B, typename C = std::less<>>
bool isLess(A a, B b, C cmp = C{}) {
  return cmp(a, b);
}

int main() {
  Eigen::Vector3d a(0, 1, 1);
  Eigen::Vector3d b(1, 1, 1);
  std::cout << std::boolalpha << (std::less<Eigen::Vector3d>{}(a, b))
            << std::endl;
  std::cout << std::boolalpha << (VectorCompare<double>{}(a, b)) << std::endl;
  std::cout << std::boolalpha << isLess(a, b, std::less<Eigen::Vector3d>())
            << std::endl;

  std::cout << std::boolalpha << (a.array() == b.array()).all() << std::endl;
  return 0;
}