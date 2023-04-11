/*
 * @Author: Poet 602289550@qq.com
 * @Date: 2023-04-10 22:34:06
 * @LastEditors: Poet 602289550@qq.com
 * @LastEditTime: 2023-04-11 20:07:41
 * @FilePath: \OBJLoader\IN_OBJ.cpp
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置
 * 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#include <Eigen\Dense>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

using V3d = Eigen::Vector3d;
using V3i = Eigen::Vector3i;

#define M_PI 3.14159265358979323846

std::vector<V3d> points;
std::vector<V3i> idx2Points;
int cnt;
double angleThreshold = 10.0;

int main() {
  std::string in_file =
      "E:\\VSProjects\\SubReconstruction\\result\\cp\\3\\cp_mesh.obj";
  std::ifstream in(in_file);
  if (!in) {
    std::cerr << "ERROR: loading obj:(" << in_file << ") file is not good"
              << std::endl;
    exit(1);
  }

  double x, y, z;
  int f0, f1, f2;
  char buffer[256] = {0};
  while (!in.getline(buffer, 255).eof()) {
    if (buffer[0] == 'v' && (buffer[1] == ' ' || buffer[1] == 32)) {
      if (sscanf_s(buffer, "v %lf %lf %lf", &x, &y, &z) == 3) {
        points.emplace_back(V3d{x, y, z});
      }
    } else if (buffer[0] == 'f' && (buffer[1] == ' ' || buffer[1] == 32)) {
      if (sscanf_s(buffer, "f %d %d %d", &f0, &f1, &f2) == 3) {
        idx2Points.emplace_back(V3i{f0 - 1, f1 - 1, f2 - 1});
      }
    }
  }

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
  }

  std::cout << "cnt = " << cnt << std::endl;
  in.close();
  return 0;
}