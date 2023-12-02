/*
 * @Author: Poet 602289550@qq.com
 * @Date: 2023-12-02 22:35:06
 * @LastEditors: Poet 602289550@qq.com
 * @LastEditTime: 2023-12-02 23:27:55
 * @FilePath: \ModelTranslateZoom\point_translate.cpp
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置
 * 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#include <Eigen/Dense>
#include <iostream>

using namespace Eigen;

int main() {
  // 使用Eigen库的MatrixXd定义3xN的矩阵，表示N个三维点云
  Eigen::MatrixXd pointCloud(3, 5); // 5个点

  // 初始化三维点云坐标
  pointCloud.col(0) << 1.0, 2.0, 3.0;
  pointCloud.col(1) << 4.0, 5.0, 6.0;
  pointCloud.col(2) << 7.0, 8.0, 9.0;
  pointCloud.col(3) << 10.0, 11.0, 12.0;
  pointCloud.col(4) << 13.0, 14.0, 15.0;

  // 打印原始点云
  std::cout << "Original Point Cloud:\n" << pointCloud << "\n\n";

  /* 1. 测试平移 */
  // 创建平移矩阵
  Eigen::Vector3d translation(1.0, 2.0, 3.0);
  // 可以使用平移变换矩阵
  Eigen::Translation3d translationMatrix(translation);
  // 也可以使用仿射变换矩阵
  Eigen::Affine3d translationAffineMatrix =
      Eigen::Affine3d::Identity(); // Eigen 的仿射矩阵
  translationAffineMatrix.translation() = translation;

  // 应用矩阵
  Eigen::MatrixXd trans_pointCloud =
      (translationAffineMatrix * pointCloud.colwise().homogeneous()).topRows(3);
  // 打印平移后的点云
  std::cout << "Translated Point Cloud:\n" << trans_pointCloud << "\n\n";

  /* 2. 测试缩放 */
  // 创建缩放矩阵
  Eigen::Vector3d scaling(2.0, 0.5, 1.5); // 缩放因子分别为2、0.5、1.5
  // 三种写法：
  // 第一种使用对角矩阵
  Eigen::DiagonalMatrix<double, 3> scaleMatrix(scaling);
  // 第二种使用AlignedScaling3d
  Eigen::AlignedScaling3d scaleAlignMatrix;
  scaleAlignMatrix.diagonal() = scaling;
  // 第三种使用仿射变换矩阵
  Eigen::Affine3d scaleAffineMatrix =
      Eigen::Affine3d::Identity(); // Eigen 的仿射矩阵
  scaleAffineMatrix.linear() = scaling.asDiagonal();

  // 应用缩放矩阵
  // Eigen::MatrixXd scale_pointCloud = scaleMatrix * pointCloud;
  // Eigen::MatrixXd scale_pointCloud = scaleAlignMatrix * pointCloud;
  Eigen::MatrixXd scale_pointCloud =
      (scaleAffineMatrix * pointCloud.colwise().homogeneous()).topRows(3);
  // 打印缩放后的点云
  std::cout << "Scaled Point Cloud:\n" << scale_pointCloud << "\n\n";

  /* 3 测试先缩放再平移 */
  // 生成仿射变换矩阵(Affine3d默认是平移-旋转-缩放，不可改动，无法达到先平移再缩放的目的)
  Eigen::Affine3d transform_affine_1 = Eigen::Affine3d::Identity();
  transform_affine_1.linear() = scaling.asDiagonal();
  transform_affine_1.translation() = translation;

  // 应用仿射变换矩阵
  Eigen::MatrixXd transform_pointCloud_1 =
      (transform_affine_1 * pointCloud.colwise().homogeneous()).topRows(3);
  // 打印先缩放再平移后的点云
  std::cout << "Transformed Point Cloud(Scale then Translation):\n"
            << transform_pointCloud_1 << "\n\n";

  /* 4 测试先平移再缩放 */
  // 只能通过矩阵乘法实现
  Eigen::MatrixXd transform_pointCloud_2 =
      (scaleAlignMatrix * translationAffineMatrix *
       pointCloud.colwise().homogeneous())
          .topRows(3);
  // 打印先平移再缩放后的点云
  std::cout << "Transformed Point Cloud(Translation then Scale):\n"
            << transform_pointCloud_2 << "\n\n";

  return 0;
}