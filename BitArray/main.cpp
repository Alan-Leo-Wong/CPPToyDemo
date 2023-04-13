/*
 * @Author: Poet 602289550@qq.com
 * @Date: 2023-04-13 16:01:24
 * @LastEditors: Poet 602289550@qq.com
 * @LastEditTime: 2023-04-13 16:22:59
 * @FilePath: \BitArray\main.cpp
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置
 * 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#include <iostream>

using namespace std;

/*bit数组：如果有8位(char类型，下面的例子是unsigned
 * int，有32位)，则可表示0~7这8个数，哪一位为1就代表这一位的数字插入进来了
 */

// 将一个整数添加到二进制数组中
void add_to_bitarray(unsigned int *bitArr,
                     int num) { /* num代表要插进数组中的数 */
  // 具体是第几个32位组由 num / 32 得到，
  int groupIdx = num >> 5;
  // 在组内的偏移位置由 num % 32 = num & 31 得到
  int bitPos = num & 31;
  // 1 << bitPos 代表 bitPos 这个位置 1，代表插入了
  bitArr[groupIdx] |= (1 << bitPos);
}

// 判断一个整数num是否在二进制数组中，若存在则返回1 << (num & 31)
int is_in_bitarray(unsigned int *bitArr, int num) {
  return bitArr[num >> 5] & (1 << (num & 31));
}

// 删除二进制数组中的一个整数
void clear_bitarray(char *bitArr, int num) {
  bitArr[num >> 5] &= ~(1 << (num & 31));
}

int main() {
  int numArray = 64;

  // unsigned int有32位，所以只需要分配 numArray / 32 bytes即可，
  unsigned int *bitArr = new unsigned int(numArray / 32);

  // 插入 num 这个数
  add_to_bitarray(bitArr, 3);
  add_to_bitarray(bitArr, 33);
  add_to_bitarray(bitArr, 32);

  cout << is_in_bitarray(bitArr, 3) << endl;  // 8
  cout << is_in_bitarray(bitArr, 33) << endl; // 2
  cout << is_in_bitarray(bitArr, 32) << endl; // 1

  cout << is_in_bitarray(bitArr, 7) << endl; // 0

  return 0;
}
