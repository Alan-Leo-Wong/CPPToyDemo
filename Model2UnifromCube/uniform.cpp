/*
 * @Author: Poet 602289550@qq.com
 * @Date: 2023-04-10 22:34:06
 * @LastEditors: Poet 602289550@qq.com
 * @LastEditTime: 2023-04-14 16:56:58
 * @FilePath: \OBJLoader\IN_OBJ.cpp
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置
 * 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#include <Eigen\Dense>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <igl\readOFF.h>
#include <igl\readPLY.h>
#include <igl\writeOFF.h>
#include <igl\writePLY.h>
#include <iostream>
#include <sstream>
#include <vector>

using namespace Eigen;

class Transformer {
public:
  Transformer(MatrixXd V, double scaleFactor_) : scaleFactor(scaleFactor_) {
    transMat = CalcTransformMatrix(V);
  }

  void Model2UnitCube(MatrixXd &V);
  void UnitCube2Model(MatrixXd &V);

private:
  Matrix4d transMat;
  double scaleFactor;

private:
  Matrix4d CalcTransformMatrix(MatrixXd V);
};

Matrix4d Transformer::CalcTransformMatrix(MatrixXd V) {
  Eigen::RowVector3d boxMin = V.colwise().minCoeff();
  Eigen::RowVector3d boxMax = V.colwise().maxCoeff();

  // Get the target solveRes (along the largest dimension)
  double scale = boxMax[0] - boxMin[0];
  double minScale = scale;
  for (int d = 1; d < 3; d++) {
    scale = std::max<double>(scale, boxMax[d] - boxMin[d]);
    minScale = std::min<double>(scale, boxMax[d] - boxMin[d]);
  }
  // std::cout << 1.1 + scale / minScale << std::endl;
  // scaleFactor =
  scale *= scaleFactor;
  Eigen::Vector3d center = 0.5 * boxMax + 0.5 * boxMin;

  for (int i = 0; i < 3; i++)
    center[i] -= scale / 2;
  Eigen::MatrixXd zoomMatrix = Eigen::Matrix4d::Identity();
  Eigen::MatrixXd transMatrix = Eigen::Matrix4d::Identity();
  for (int i = 0; i < 3; i++) {
    zoomMatrix(i, i) = 1. / scale;
    transMatrix(3, i) = -center[i];
  }
  return zoomMatrix * transMatrix;
}

void Transformer::Model2UnitCube(MatrixXd &V) {
  for (int i = 0; i < V.rows(); ++i) {
    V.row(i) += transMat.block(3, 0, 1, 3);
    V.row(i) = V.row(i) * transMat.block(0, 0, 3, 3);
  }
}

void Transformer::UnitCube2Model(MatrixXd &V) {
  Eigen::Matrix3d inverseTrans = transMat.block(0, 0, 3, 3).inverse();
  for (int i = 0; i < V.rows(); ++i) {
    V.row(i) = V.row(i) * inverseTrans;
    V.row(i) -= transMat.block(3, 0, 1, 3);
  }
}

int main() {
  // const std::string in_file =
  //     "E:\\zc\\dualcontouring_uniform_model\\d4_smooth.ply";
  // const std::string out_file =
  //     "E:\\zc\\dualcontouring_uniform_model\\uniform\\d4_smooth_uniform.off";
  const std::string in_file = "E:\\zc\\mon.off";
  const std::string out_file = "E:\\zc\\origin_uniform_model\\mon_uniform.ply";
  MatrixXd V;
  MatrixXi F;
  // igl::readPLY(in_file, V, F);
  igl::readOFF(in_file, V, F);
  Transformer trans(V, 1.0);
  trans.Model2UnitCube(V);
  igl::writePLY(out_file, V, F);
  // igl::writeOFF(out_file, V, F);
  return 0;
}