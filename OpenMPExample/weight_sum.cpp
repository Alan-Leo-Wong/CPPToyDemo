/*
 * @Author: Poet 602289550@qq.com
 * @Date: 2023-05-13 10:43:56
 * @LastEditors: Poet 602289550@qq.com
 * @LastEditTime: 2023-05-13 11:17:36
 * @FilePath: \OpenMPExample\weight_sum.cpp
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置
 * 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#include "timer.hpp"
#include <algorithm>
#include <iostream>
#include <omp.h>
#include <vector>

std::vector<double> normal_weightSum(const int &m, const int &n) {
  std::vector<double> res(m);
  std::vector<double> val(n, 1);
  for (int i = 0; i < m; ++i) {
    double sum = 0;
    for (int j = 0; j < n; ++j) {
      sum += j * val[j];
    }
    res[i] = sum + i;
  }
  return res;
}

std::vector<double> mt_weightSum(const int &m, const int &n) {
  std::vector<double> res(m);
  std::vector<double> val(n, 1);
#pragma omp parallel
  for (int i = 0; i < m; ++i) {
    double sum = 0;
    // res[i] = i;
    // double &sum = res[i]; // 不能是引用
#pragma omp parallel for reduction(+ : sum)
    for (int j = 0; j < n; ++j) {
      sum += j * val[j];
      //   res[i] += j * val[j];
    }
    res[i] = sum + i;
    // sum += i;
  }
  return res;
}

std::vector<double> simd_weightSum(const int &m, const int &n) {
  std::vector<double> res(m);
  std::vector<double> val(n, 1);
#pragma omp parallel
  for (int i = 0; i < m; ++i) {
    double sum = 0;
// res[i] = i;
#pragma omp simd simdlen(8)
    for (int j = 0; j < n; ++j) {
      sum += j * val[j];
      //   res[i] += j * val[j];
    }
    res[i] = sum + i;
  }
  return res;
}

constexpr int m = 100000;
constexpr int n = 20;

int main() {
  std::vector<double> normal_res;
  std::vector<double> mt_res;
  std::vector<double> simd_res;

  const auto &normal_tictoc = [&]() {
    static double t_start = get_seconds();
    normal_res = mt_weightSum(m, n);
    double diff = get_seconds() - t_start;
    t_start += diff;
    return diff;
  };

  const auto &mt_tictoc = [&]() {
    static double t_start = get_seconds();
    mt_res = mt_weightSum(m, n);
    double diff = get_seconds() - t_start;
    t_start += diff;
    return diff;
  };

  const auto &simd_tictoc = [&]() {
    static double t_start = get_seconds();
    simd_res = simd_weightSum(m, n);
    double diff = get_seconds() - t_start;
    t_start += diff;
    return diff;
  };

  std::cout << "normal time: " << normal_tictoc() << " s" << std::endl;
  std::cout << "multi-thread time: " << mt_tictoc() << " s" << std::endl;
  std::cout << "SIMD time: " << simd_tictoc() << " s" << std::endl;

  for (int i = 0; i < m; ++i) {
    if (normal_res[i] != mt_res[i])
      std::cout << "i = " << i << ", normal is not equal mt\n";
    if (normal_res[i] != simd_res[i])
      std::cout << "i = " << i << ", normal is not equal simd\n";
  }
  return 0;
}