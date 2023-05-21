/*
 * @Author: Poet 602289550@qq.com
 * @Date: 2023-04-10 22:34:06
 * @LastEditors: Poet 602289550@qq.com
 * @LastEditTime: 2023-05-21 15:31:06
 * @FilePath: \OBJLoader\IN_OBJ.cpp
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置
 * 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#include <Eigen\Dense>
#include <cmath>
#include <cstdio>
#include <fstream>
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
  // std::string in_file =
  //     "E:\\VSProjects\\SubReconstruction\\result\\cp\\3\\cp_mesh.obj";
  // std::string in_file = "E:\\temp\\pkc_mesh.obj"; // ours
  // std::string in_file = "E:\\temp\\cgal\\elk_CGAL.obj"; // cgal
  // std::string in_file = "E:\\temp\\gd\\pkc-gd.obj"; // gd
  // std::string in_file = "E:\\temp\\mc\\pkc-mc.obj"; // mc
  std::string in_file = "E:\\zc\\model\\pkc_mesh_rdt.obj"; // rdt
  std::ifstream in(in_file);
  if (!in) {
    std::cerr << "ERROR: loading obj:(" << in_file << ") file is not good"
              << std::endl;
    exit(1);
  }

  double x, y, z;
  int f0, _f0, f1, _f1, f2, _f2;
  char buffer[256] = {0};
  while (!in.getline(buffer, 255).eof()) {
    if (buffer[0] == 'v' && (buffer[1] == ' ' || buffer[1] == 32)) {
      if (sscanf_s(buffer, "v %lf %lf %lf", &x, &y, &z) == 3) {
        points.emplace_back(V3d{x, y, z});
      }
    } else if (buffer[0] == 'f' && (buffer[1] == ' ' || buffer[1] == 32)) {
      if (sscanf_s(buffer, "f %d//%d %d//%d %d//%d", &f0, &_f0, &f1, &_f1, &f2,
                   &_f2) == 6) {
        idx2Points.emplace_back(V3i{f0 - 1, f1 - 1, f2 - 1});
      } else if (sscanf_s(buffer, "f %d %d %d", &f0, &f1, &f2) == 3) {
        idx2Points.emplace_back(V3i{f0 - 1, f1 - 1, f2 - 1});
      }
    }
  }

  double prop[5] = {0};
  const int triFaces = idx2Points.size();
  for (const auto &idx : idx2Points) {
    std::vector<V3d> tri;
    tri.emplace_back(points[idx(0)]);
    tri.emplace_back(points[idx(1)]);
    tri.emplace_back(points[idx(2)]);

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

    double G = cpQuality(points[idx(0)], points[idx(1)], points[idx(2)]);
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
      << ", percentage = " << (cnt * 100.0 / triFaces) << "%" << std::endl;

  std::cout << "-- G_min = " << minG << std::endl;
  std::cout << "-- G_avg = " << avgG << std::endl;
  // std::cout << "-- G_area = " << minG << std::endl;
  std::cout << "-- Thtea_min = " << minTheta << std::endl;
  std::cout << "-- Thtea_avg = " << avgMinTheta << std::endl;

  in.close();
  return 0;
}