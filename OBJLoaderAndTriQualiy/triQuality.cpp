#include <Eigen\Dense>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <igl/read_triangle_mesh.h>
#include <iostream>
#include <limits>
#include <sstream>
#include <vector>

using V3d = Eigen::Vector3d;
using V3i = Eigen::Vector3i;

#define M_PI 3.14159265358979323846

std::vector<V3d> points;
std::vector<V3i> idx2Points;
int cnt;
double angleThreshold = 30.0;
double avgG = .0;
double minG = std::numeric_limits<double>::max();
double minTheta = std::numeric_limits<double>::max();
double avgMinTheta = .0;

double cpSquare(const V3d &p0, const V3d &p1, const V3d &p2) {
  return 0.5 * (((p1 - p0).cross(p2 - p0)).norm());
}

double cpCircum(const V3d &p0, const V3d &p1, const V3d &p2) {
  return (p1 - p0).norm() + (p2 - p1).norm() + (p0 - p2).norm();
}

double cpQuality(const V3d &p0, const V3d &p1, const V3d &p2) {
  double coef = 6 / std::sqrt(3);
  double St = cpSquare(p0, p1, p2);
  double totalArea = .0;
  totalArea += St;
  double pt = cpCircum(p0, p1, p2) * 0.5;

  double ht =
      std::max((p1 - p0).norm(), std::max((p2 - p1).norm(), (p0 - p2).norm()));
  return coef * St / (pt * ht);
}

int main() {
  Eigen::MatrixXd V;
  Eigen::MatrixXi F;
  std::string in_file = "E:\\VSProjects\\3DThinShell\\model\\switchmec.obj";
  igl::read_triangle_mesh(in_file, V, F);

  for (int k = 0; k < F.rows(); ++k) {
    std::vector<V3d> tri;
    V3d p0 = V.row(F.row(k)[0]);
    V3d p1 = V.row(F.row(k)[1]);
    V3d p2 = V.row(F.row(k)[2]);
    tri.emplace_back(p0);
    tri.emplace_back(p1);
    tri.emplace_back(p2);

    double minAngle = 180;
    for (int i = 0; i < 3; ++i) {
      V3d ed_0 = tri[i], ed_1 = tri[(i + 1) % 3]; // 一条边的两个顶点
      V3d edge_0 = ed_1 - ed_0;

      ed_1 = tri[(i + 2) % 3];
      V3d edge_1 = ed_1 - ed_0;

      double cosAngle =
          edge_0.dot(edge_1) / (edge_0.norm() * edge_1.norm()); // 角度cos值
      double radiAngle = acos(cosAngle) * 180 / M_PI;           // 弧度角

      minAngle = std::min(minAngle, radiAngle);
      // std::cout << minAngle << std::endl;
    }
    if (minAngle < angleThreshold)
      ++cnt;

    double G = cpQuality(p0, p1, p2);
    avgG += G;
    minG = std::min(minG, G);

    minTheta = std::min(minTheta, minAngle);
    avgMinTheta += minTheta;

    // if (0 < G && G <= 0.2)
    //   prop[0]++;
    // else if (0.2 < G && G <= 0.4)
    //   prop[1]++;
    // else if (0.4 < G && G <= 0.6)
    //   prop[2]++;
    // else if (0.6 < G && G <= 0.8)
    //   prop[3]++;
    // else if (0.8 < G && G <= 1)
    //   prop[4]++;
  }

  // for (int i = 0; i < 5; ++i) {
  //   prop[i] /= triFaces;
  //   std::cout << prop[i] << std::endl;
  // }

  std::cout
      << "-- The number of triangles with their minimal angles smaller than "
      << angleThreshold << " is " << cnt
      << ", percentage = " << (cnt * 100.0 / F.rows()) << "%" << std::endl;

  std::cout << "-- G_min = " << minG << std::endl;
  std::cout << "-- G_avg = " << avgG / F.rows() << std::endl;
  // std::cout << "-- G_area = " << minG << std::endl;
  std::cout << "-- Thtea_min = " << minTheta << std::endl;
  std::cout << "-- Thtea_avg = " << avgMinTheta * 3 / F.rows() << std::endl;

  return 0;
}
