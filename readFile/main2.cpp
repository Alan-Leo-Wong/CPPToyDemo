/*
 * @Author: Poet 602289550@qq.com
 * @Date: 2023-03-17 21:05:08
 * @LastEditors: Poet 602289550@qq.com
 * @LastEditTime: 2023-03-27 15:44:50
 * @FilePath: \readFile\main2.cpp
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置
 * 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#include <algorithm>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <vector>

int main() {
  // std::ifstream in(
  //     "E:\\VSProjects\\3DThinShell\\output\\bunny\\6\\BSplineValue.txt");

  // double min_val = std::numeric_limits<double>::max();
  // double max_val = -std::numeric_limits<double>::min();

  // double t;
  // while (in >> t) {
  //   min_val = std::min(min_val, t);
  //   max_val = std::max(max_val, t);
  // }
  // std::cout << "min_val = " << min_val << "\nmax_val = " << max_val
  //           << std::endl;

  int x = 2, y = 3, z = 4;
  for (int i = 0; i < x * y * z; ++i)
    std::cout << "val = " << i << ", i = " << i % x
              << ", j = " << i % (x * y) / x << ", k = " << i / (x * y) << std::endl;

  // std::ifstream in_1(
  //     "E:\\VSProjects\\3DThinShell\\output\\bunny\\5\\Matrix_1.txt");
  // std::ifstream in_2(
  //     "E:\\VSProjects\\3DThinShell\\output\\bunny\\5\\Matrix.txt");

  // char buf_1[10240];
  // char buf_2[10240];
  // int i = 0;
  // std::vector<std::vector<double>> vec_1, vec_2;
  // while (in_1.getline(buf_1, sizeof(buf_1))) {
  //   in_2.getline(buf_2, sizeof(buf_2));

  //   std::stringstream ss_1(buf_1);
  //   std::stringstream ss_2(buf_2);

  //   int j = 0;
  //   std::vector<double> temp_1, temp_2;
  //   double val_1, val_2;

  //   while (ss_1 >> val_1) {
  //     temp_1.emplace_back(val_1);

  //     ss_2 >> val_2;
  //     temp_2.emplace_back(val_2);

  //     if (val_1 != val_2) {
  //       std::cout << "i = " << i << ", j = " << j << ", val_1 = " << val_1
  //                 << ", val_2 = " << val_2 << std::endl;
  //     }

  //     j++;
  //   }

  //   vec_1.emplace_back(temp_1);
  //   vec_2.emplace_back(temp_2);

  //   i++;
  // }
  return 0;
}