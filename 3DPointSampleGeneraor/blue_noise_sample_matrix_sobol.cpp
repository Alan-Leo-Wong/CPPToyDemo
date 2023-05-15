/*
 * @Author: Poet 602289550@qq.com
 * @Date: 2023-05-15 18:33:47
 * @LastEditors: Poet 602289550@qq.com
 * @LastEditTime: 2023-05-15 18:34:06
 * @FilePath: \3DPointSampleGeneraor\blue_noise_sample_matrix_sobol.cpp
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置
 * 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
/* 使用 sobol 序列在给定三维空间中生成均匀采样点 */
#include <cmath>
#include <iostream>
#include <vector>

// Sobol direction numbers for up to 21201 dimensions
static const unsigned int sobolDirectionNumbers[][50] = {
    // Direction numbers for 1 dimension
    {1u, 0u},
    // Direction numbers for 2 dimensions
    {1u, 1u, 0u},
    // Direction numbers for 3 dimensions
    {1u, 3u, 1u, 0u},
    // Direction numbers for 4 dimensions
    {1u, 7u, 5u, 1u, 0u},
    // ... direction numbers for higher dimensions
};

// Helper function to calculate the number of bits required to represent a value
unsigned int countBits(unsigned int value) {
  unsigned int count = 0;
  while (value > 0) {
    value >>= 1;
    count++;
  }
  return count;
}

// Helper function to calculate the ith component of the Sobol sequence
float sobolSample(unsigned int i, unsigned int n) {
  unsigned int bits = countBits(n);
  unsigned int directionCount =
      sizeof(sobolDirectionNumbers) / sizeof(sobolDirectionNumbers[0]);

  if (i >= directionCount) {
    std::cerr << "Sobol sequence not supported for " << i << " dimensions."
              << std::endl;
    return 0.0f;
  }

  const unsigned int *directionNumbers = sobolDirectionNumbers[i];
  unsigned int directionCountBits = countBits(directionNumbers[0]);

  if (bits > directionCountBits) {
    std::cerr << "Sobol sequence not supported for " << bits << " bits."
              << std::endl;
    return 0.0f;
  }

  unsigned int result = 0;
  for (unsigned int j = 1; j <= bits; ++j) {
    if ((n & (1u << (bits - j))) != 0) {
      result ^= directionNumbers[j];
    }
  }

  return static_cast<float>(result) / static_cast<float>(1u << bits);
}

// Generate uniform samples in the given 3D space using Sobol sequence
std::vector<std::vector<float>>
generateSobolSamples(unsigned int numSamples, float minArea, float maxArea) {
  unsigned int numDimensions = 3;
  unsigned int currentSample = 0;

  std::vector<std::vector<float>> samples;
  samples.reserve(numSamples);

  while (currentSample < numSamples) {
    std::vector<float> sample(numDimensions);
    for (unsigned int i = 0; i < numDimensions; ++i) {
      sample[i] = minArea + sobolSample(i, currentSample);
      sample[i] *= maxArea - minArea;
    }
    samples.push_back(sample);
    currentSample++;
  }

  return samples;
}

int main() {
  unsigned int numSamples = 10;
  float minArea = 0.0f;
  float maxArea = 1.0f;

  std::vector<std::vector<float>> samples =
      generateSobolSamples(numSamples, minArea, maxArea);

  for (const auto &sample : samples) {
    std::cout << "Sample: ";
    for (float value : sample) {
      std::cout << value << " ";
    }
    std::cout << std::endl;
  }

  return 0;
}
