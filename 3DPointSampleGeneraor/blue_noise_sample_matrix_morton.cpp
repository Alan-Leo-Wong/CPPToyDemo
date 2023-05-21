/* 使用蒙特卡洛方法在给定三维空间中生成均匀采样点 */
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

// 定义一个表示三维向量的结构体
struct Vector3D {
  double x, y, z;
};

// 生成介于[min, max]之间的随机浮点数
double randomDouble(double min, double max) {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dis(min, max);
  return dis(gen);
}

// 计算两个三维向量之间的距离
double distance(const Vector3D &v1, const Vector3D &v2) {
  double dx = v1.x - v2.x;
  double dy = v1.y - v2.y;
  double dz = v1.z - v2.z;
  return std::sqrt(dx * dx + dy * dy + dz * dz);
}

// 检查点是否满足蓝噪声采样的条件
bool isBlueNoise(const Vector3D &point, const std::vector<Vector3D> &samples,
                 double minDistance) {
  for (const auto &sample : samples) {
    if (distance(sample, point) < minDistance) {
      return false;
    }
  }
  return true;
}

// 在[minArea, maxArea]范围内进行蓝噪声采样
std::vector<Vector3D> generateBlueNoiseSamples(const Vector3D &minArea,
                                               const Vector3D &maxArea,
                                               int numSamples,
                                               double minDistance) {
  std::vector<Vector3D> samples;

  while (samples.size() < numSamples) {
    Vector3D point;
    point.x = randomDouble(minArea.x, maxArea.x);
    point.y = randomDouble(minArea.y, maxArea.y);
    point.z = randomDouble(minArea.z, maxArea.z);

    if (isBlueNoise(point, samples, minDistance)) {
      samples.push_back(point);
    }
  }

  return samples;
}

int main() {
  Vector3D minArea{0.0, 0.0, 0.0}; // 最小区域的顶点
  Vector3D maxArea{1.0, 1.0, 1.0}; // 最大区域的顶点
  int numSamples = 100000;         // 采样点的数量
  double minDistance = 0.1;        // 最小距离

  std::vector<Vector3D> blueNoiseSamples =
      generateBlueNoiseSamples(minArea, maxArea, numSamples, minDistance);

  // 输出采样点的坐标
  //   for (const auto &sample : blueNoiseSamples) {
  //     std::cout << "X: " << sample.x << ", Y: " << sample.y << ", Z: " <<
  //     sample.z
  //               << std::endl;
  //   }

  std::ofstream out(".\\blue_noise_points.xyz");
  for (const auto &sample : blueNoiseSamples) {
    out << sample.x << " " << sample.y << " " << sample.z << std::endl;
  }

  return 0;
}