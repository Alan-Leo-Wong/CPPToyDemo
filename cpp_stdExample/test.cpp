/*
 * @Author: WangLei
 * @authorEmail: leiw1006@gmail.com
 * @Date: 2023-05-05 10:42:33
 * @LastEditors: WangLei
 * @LastEditTime: 2023-05-05 10:44:32
 * @FilePath: \cpp_stdExample\test.cpp
 * @Description:
 */
#include "test.h"
#include <chrono>

inline double igl::get_seconds() {
  return std::chrono::duration<double>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}
