/*
 * @Author: Lei Wang 602289550@qq.com
 * @Date: 2023-03-09 13:56:39
 * @LastEditors: Lei Wang 602289550@qq.com
 * @LastEditTime: 2023-03-09 16:33:55
 * @FilePath: \ToyDemo\3slabs\main.cpp
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置
 * 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#include <Eigen/Dense>
#include <algorithm>
#include <iostream>
#include <limits>
#include <stdlib.h>
#include <vector>

using V3d = Eigen::Vector3d;
using Edge = std::pair<V3d, V3d>;
const double DINF = std::numeric_limits<double>::max();

struct Node {
  V3d backPoint;
  V3d frontPoint;
  V3d width;

  Node() {}
  Node(const V3d &_recBackPoint, const V3d &_recFrontPoint, const V3d &_width)
      : backPoint(_recBackPoint), frontPoint(_recFrontPoint), width(_width) {}
};

void initialLine(const V3d &v1, const V3d &v2, Edge &line) {
  line = std::make_pair(v1, v2);
}

template <typename T>
bool isInRange(const double &l, const double &r, const T &query) {
  return l <= query && query <= r;
}

template <typename... T>
bool isInRange(const double &l, const double &r, const T &...query) {
  return isInRange(query...);
}

void slabForIntersection(const Node &node, const Edge &line,
                         std::vector<V3d> &res) {
  V3d p1 = line.first;
  V3d p2 = line.second;
  // p2 = p1 + t * dir
  double t = (p2 - p1).norm();
  V3d dir = p2 - p1;

  V3d recBackPoint = node.backPoint;
  V3d width = node.width;

  // bottom plane
  double bottom_t = DINF;
  if (dir.z() != 0) // 平面法线与line垂直，说明line与平面平行
    bottom_t = (recBackPoint.z() - p1.z()) / dir.z();
  // left plane
  double left_t = DINF;
  if (dir.y() != 0) // 平面法线与line垂直，说明line与平面平行
    left_t = (recBackPoint.y() - p1.y()) / dir.y();
  // back plane
  double back_t = DINF;
  if (dir.x() != 0) // 平面法线与line垂直，说明line与平面平行
    back_t = (recBackPoint.x() - p1.x()) / dir.x();

  //   if (isInRange(.0, 1.0, bottom_t, left_t, back_t)) {
  //   }

  if (isInRange(.0, 1.0, bottom_t))
    res.emplace_back(p1 + bottom_t * dir);
  if (isInRange(.0, 1.0, left_t))
    res.emplace_back(p1 + left_t * dir);
  if (isInRange(.0, 1.0, back_t))
    res.emplace_back(p1 + back_t * dir);
}

int main() {
  Edge line;
  initialLine(V3d(0, 0, -1), V3d(0, 2, 1), line);

  V3d recBackPoint = V3d(-1, -1, 0);
  V3d recFrontPoint = V3d(1, 1, 2);
  V3d width = V3d(2, 2, 2);
  Node node(recBackPoint, recFrontPoint, width);
  std::vector<V3d> res;

  slabForIntersection(node, line, res);
  for (const auto &inter : res)
    std::cout << inter << std::endl;

  return EXIT_SUCCESS;
}
