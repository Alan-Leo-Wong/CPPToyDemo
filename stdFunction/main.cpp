/*
 * @Author: WangLei
 * @authorEmail: leiw1006@gmail.com
 * @Date: 2023-04-30 15:32:53
 * @LastEditors: WangLei
 * @LastEditTime: 2023-04-30 15:36:04
 * @FilePath: \stdFunction\main.cpp
 * @Description:
 */
#include <algorithm>
#include <functional>
#include <iostream>
#include <iterator>
#include <numeric>
#include <vector>

int main() {
  std::vector<int> vec(2, 1);
  std::exclusive_scan(vec.begin(), vec.end(), vec.begin(), 0);
  for (int i = 0; i < vec.size(); ++i)
    std::cout << vec[i] << std::endl;
  return 0;
}