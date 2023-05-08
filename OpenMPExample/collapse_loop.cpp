/*
 * @Author: Poet 602289550@qq.com
 * @Date: 2023-05-08 19:50:24
 * @LastEditors: Poet 602289550@qq.com
 * @LastEditTime: 2023-05-08 20:33:37
 * @FilePath: \OpenMPExample\collapse_loop.cpp
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置
 * 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#include <algorithm>
#include <chrono>
#include <iostream>
#include <iterator>
#include <omp.h>
#include <vector>

double get_seconds() {
  return std::chrono::duration<double>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

int main() {
  std::vector<int> vec;

  const auto &tictoc = [&]() {
    static double t_start = get_seconds();

// 乱序
#pragma omp parallel
    for (int i = 0; i < 10000; ++i) {
      std::vector<int> vec_private;
#pragma omp for nowait // fill vec_private in parallel
      for (int x = 0; x < 100; x++)
        for (int y = 0; y < 10; y++)
          for (int z = 0; z < 10; z++)
            vec_private.push_back(x * 10000 + y * 100 + z);
#pragma omp critical
      vec.insert(vec.end(), vec_private.begin(), vec_private.end());
    }

    double diff = get_seconds() - t_start;
    t_start += diff;
    return diff;
  };
  std::cout << "part 1: " << tictoc() << " s" << std::endl;
  std::cout << vec.size() << std::endl;

  //   std::copy(vec.begin(), vec.end(), std::ostream_iterator<int>(std::cout, "
  //   "));
  return 0;
}