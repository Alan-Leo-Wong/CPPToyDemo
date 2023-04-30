/*
 * @Author: Poet 602289550@qq.com
 * @Date: 2023-04-27 19:57:30
 * @LastEditors: Poet 602289550@qq.com
 * @LastEditTime: 2023-04-27 19:57:36
 * @FilePath: \BitArray\main2.cpp
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置
 * 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#include <iostream>

using namespace std;

class Age {
public:
  Age &operator++() // 前置++
  {
    ++i;
    return *this;
  }

  const Age operator++(int) // 后置++
  {
    Age tmp = *this;
    ++(*this); // 利用前置++
    return tmp;
  }

  Age &operator=(int i) // 赋值操作
  {
    this->i = i;
    return *this;
  }

private:
  int i;
};

int main() { return 0; }
