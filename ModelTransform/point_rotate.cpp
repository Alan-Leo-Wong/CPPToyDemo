#include <Eigen/Dense>
#include <Eigen/src/Core/Matrix.h>
#include <iostream>

using namespace Eigen;

#define M_PI 3.145926

int main() {
  // 旋转向量（也称等效旋转矢量）使用 AngleAxis,
  // AngleAxis(angle_in_radian,Vector3f(ax,ay,az)) The axis vector must be
  // normalized. 旋转都是逆时针为正
  // 它的底层不直接是
  // Matrix，但运算可以当作矩阵（因为重载了运算符）且可以转成矩阵
  AngleAxisd rotation_vector(
      M_PI / 4,
      Vector3d::UnitY()); // 沿 y 轴旋转 45 度，使用弧度制，Vector3d::UnitY()
                          // y轴单位向量
  // 将等效矢量转换为旋转矩阵
  Matrix3d rotate_mat = rotation_vector.toRotationMatrix();

  // 旋转矩阵转换为欧拉角
  Vector3d RPY_angle = rotate_mat.eulerAngles(
      0, 1,
      2); // XYZ顺序，即roll-pitch-yaw顺序(绕X、Y、Z轴旋转的姿态角分别为roll、pitch、yaw，旋转顺序为Z->Y->X)。

  std::cout << "rotation_matrixr:\n" << rotation_vector.matrix() << "\n\n";
  std::cout << "rotation_matrix:\n" << rotate_mat << "\n\n"; // 结果和上面一样
  std::cout << "roll pitch yaw:\n" << RPY_angle.transpose() << "\n==========\n";

  /* 由欧拉角得到旋转矩阵 */
  double yaw = 10.0, pitch = 20.0, roll = 50.0;
  Eigen::AngleAxis Vz(yaw, Eigen::Vector3d::UnitZ());
  Eigen::AngleAxis Vy(pitch, Eigen::Vector3d::UnitY());
  Eigen::AngleAxis Vx(roll, Eigen::Vector3d::UnitX());

  //   Eigen::AngleAxis t_V =
  //       Vz * Vy * Vx; // error: Eigen::AngleAxis 不能在声明的同时赋值
  Eigen::Matrix3d t_V(Vz * Vy * Vx);
  std::cout << "rotation matrix:\n" << t_V << "\n==========\n";

  /* 使用仿射变换矩阵实现旋转变换 */
  Eigen::MatrixXd pointCloud(3, 5);
  // 初始化三维点云坐标
  pointCloud.col(0) << 1.0, 2.0, 3.0;
  pointCloud.col(1) << 4.0, 5.0, 6.0;
  pointCloud.col(2) << 7.0, 8.0, 9.0;
  pointCloud.col(3) << 10.0, 11.0, 12.0;
  pointCloud.col(4) << 13.0, 14.0, 15.0;

  // 创建旋转变换
  double angleInDegrees = 45.0; // 旋转角度为45度
  Eigen::Affine3d rotation = Eigen::Affine3d::Identity();
  rotation.rotate(Eigen::AngleAxisd(angleInDegrees * M_PI / 180.0,
                                    Eigen::Vector3d::UnitZ()));

  // 应用旋转变换
  Eigen::MatrixXd rotate_pointCloud =
      (rotation * pointCloud.colwise().homogeneous()).topRows(3);
  // 打印旋转后的点云
  std::cout << "Rotated Point Cloud:\n"
            << rotate_pointCloud << "\n==========\n";

  return 0;
}