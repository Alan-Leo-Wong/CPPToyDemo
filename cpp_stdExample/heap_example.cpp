/*
 * @Author: Alan Wang leiw1006@gmail.com
 * @Date: 2023-12-13 21:45:37
 * @LastEditors: Alan Wang leiw1006@gmail.com
 * @LastEditTime: 2023-12-13 22:48:27
 * @FilePath: \cpp_stdExample\heap_example.cpp
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置
 * 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#include <iostream>
#include <queue>
#include <set>
#include <vector>

template <typename T> struct IndexedValue {
  int index;
  T value;

  IndexedValue() = default;

  IndexedValue(int idx, const T &val) : index(idx), value(val) {}

  // 按照值进行升序排序
  static bool asc_cmp(const IndexedValue<T> &a, const IndexedValue<T> &b) {
    return a.value < b.value;
  }

  // 按照值进行降序排序
  static bool des_cmp(const IndexedValue<T> &a, const IndexedValue<T> &b) {
    return a.value > b.value;
  }

  template <typename T1>
  friend std::ostream &operator<<(std::ostream &os, const IndexedValue<T1> &iv);
};

template <typename T1>
std::ostream &operator<<(std::ostream &os, const IndexedValue<T1> &iv) {
  os << "index = " << iv.index << ", value = " << iv.value;
  return os;
}

// // 按照值进行升序排序
// template <typename T>
// bool asc_cmp(const IndexedValue<T> &a, const IndexedValue<T> &b) {
//   return a.value < b.value;
// }

// // 按照值进行降序排序
// template <typename T>
// bool des_cmp(const IndexedValue<T> &a, const IndexedValue<T> &b) {
//   return a.value > b.value;
// }

int main() {
  /* 注意 std::priority_queue 不会在数据更新后重新排序 */

  std::vector<IndexedValue<int>> in_data;
  in_data.reserve(4);
  in_data.emplace_back(IndexedValue<int>(0, 0));
  in_data.emplace_back(IndexedValue<int>(1, 1));
  in_data.emplace_back(IndexedValue<int>(2, 2));
  in_data.emplace_back(IndexedValue<int>(3, 3));

  std::priority_queue<IndexedValue<int>, std::vector<IndexedValue<int>>,
                      decltype(&IndexedValue<int>::des_cmp)>
      minHeap(&IndexedValue<int>::des_cmp);
  for (const auto &data : in_data) {
    minHeap.push(data);
  }

  std::cout << "Min Heap:" << std::endl;
  while (!minHeap.empty()) {
    std::cout << minHeap.top() << std::endl;
    minHeap.pop();
  }

  std::cout << "==========\n";

  std::priority_queue<IndexedValue<int>, std::vector<IndexedValue<int>>,
                      decltype(&IndexedValue<int>::asc_cmp)>
      maxHeap(&IndexedValue<int>::asc_cmp);
  for (const auto &data : in_data) {
    maxHeap.push(data);
  }
  std::cout << "Max Heap:" << std::endl;
  while (!maxHeap.empty()) {
    std::cout << maxHeap.top() << std::endl;
    maxHeap.pop();
  }

  std::cout << "==========\n";

  /* 想要达到在数据更新后重新排序的效果, 可以使用 set 的快速删除与重新插入来实现
   */
  std::set<IndexedValue<int>, decltype(&IndexedValue<int>::des_cmp)> maxSet(
      IndexedValue<int>::des_cmp);
  maxSet.insert(in_data.begin(), in_data.end());
  std::cout << "Max Set(Original Data):" << std::endl;
  for (const auto &data : maxSet) {
    std::cout << data << std::endl;
  }
  std::cout << "==========\n";
  // 修改元素值
  auto it = maxSet.find(IndexedValue<int>(0, 0));
  if (it != maxSet.end()) {
    maxSet.erase(it);
    maxSet.insert(IndexedValue<int>(0, 4));
  }
  std::cout << "Max Set(Modified Data):" << std::endl;
  for (const auto &data : maxSet) {
    std::cout << data << std::endl;
  }
  return 0;
}