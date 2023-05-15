/* 使用 halton 序列在给定三维空间中生成均匀采样点 */
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

// 定义一个表示三维向量的结构体
struct Vector3D {
  double x, y, z;
};

// 生成Halton序列的第index个值
double haltonSequence(int index, int base) {
  double result = 0.0;
  double f = 1.0 / base;
  int i = index;

  while (i > 0) {
    result += f * (i % base);
    i = std::floor(i / base);
    f /= base;
  }

  return result;
}

// 将Halton序列值映射到[min, max]范围内
double mapToRange(double value, double min, double max) {
  return min + value * (max - min);
}

// 在[minArea, maxArea]范围内进行蓝噪声采样
std::vector<Vector3D> generateBlueNoiseSamples(const Vector3D &minArea,
                                               const Vector3D &maxArea,
                                               int numSamples) {
  std::vector<Vector3D> samples;
  int baseX = 2; // X轴上的基数
  int baseY = 3; // Y轴上的基数
  int baseZ = 5; // Z轴上的基数

  for (int i = 0; i < numSamples; ++i) {
    double x = mapToRange(haltonSequence(i, baseX), minArea.x, maxArea.x);
    double y = mapToRange(haltonSequence(i, baseY), minArea.y, maxArea.y);
    double z = mapToRange(haltonSequence(i, baseZ), minArea.z, maxArea.z);
    samples.push_back({x, y, z});
  }

  return samples;
}

int main() {
  Vector3D minArea{-10.0, -20.0, -30.0}; // 最小区域的顶点
  Vector3D maxArea{5.0, 3.0, 8.0};       // 最大区域的顶点
  int numSamples = 1000000;              // 采样点的数量

  std::vector<Vector3D> blueNoiseSamples =
      generateBlueNoiseSamples(minArea, maxArea, numSamples);

  // 输出采样点的坐标
  // for (const auto &sample : blueNoiseSamples) {
  //   std::cout << "X: " << sample.x << ", Y: " << sample.y << ", Z: " <<
  //   sample.z
  //             << std::endl;
  // }

  std::ofstream out(".\\blue_noise_points_halton.xyz");
  for (const auto &sample : blueNoiseSamples) {
    out << sample.x << " " << sample.y << " " << sample.z << std::endl;
  }
  return 0;
}