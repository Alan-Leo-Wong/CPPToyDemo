/*
 * @Author: Alan Wang leiw1006@gmail.com
 * @Date: 2023-12-08 23:15:23
 * @LastEditors: Alan Wang leiw1006@gmail.com
 * @LastEditTime: 2023-12-08 23:26:47
 * @FilePath: \EigenExample\2Dand3D.cpp
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置
 * 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#include <Eigen/Dense>
#include <iostream>

using namespace Eigen;

int main() {
  Vector3d a = Vector3d(-1, 0, 2);
  Vector3d b = Vector3d(1, 0, 2);
  Vector3d c = Vector3d(0, 2, 2);

  Vector3d n = (b - a).cross(c - a).normalized();

  Vector3d o = a;
  Vector3d x = (b - o).normalized();
  Vector3d y = n.cross(x).normalized();

  // p - o由x y n张成
  Vector3d p = Vector3d(0, 1, 2) - o;
  Matrix3d m;
  m.col(0) = x;
  m.col(1) = y;
  m.col(2) = n;

  Vector3d local = m.inverse() * p;

  std::cout << local.transpose() << std::endl;

  Vector2d x_2 = (b - o).head<2>().normalized();
  Vector2d y_2 = n.cross(x).head<2>().normalized();

  // p_2 - o_2由x_2 y_2张成
  Vector2d p_2 = Vector2d(0, 1) - o.head<2>();
  Matrix2d m_2;
  m_2.col(0) = x_2;
  m_2.col(1) = y_2;
  Vector2d local_2 = m_2.inverse() * p_2;
  std::cout << local_2.transpose() << std::endl;

  return 0;
}