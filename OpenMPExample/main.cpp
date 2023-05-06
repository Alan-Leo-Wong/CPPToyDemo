#include <algorithm>
#include <iostream>
#include <omp.h>
#include <vector>

#pragma omp declare reduction(                                                 \
    merge: std::vector<int>                                                                 \
    : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))

int main() {
  // std::vector<int> vec;

  // #pragma omp declare reduction (reduction-identifier : typename-list :
  // combiner) [initializer-clause]
  /* reduction-identifier：归约标识符，相当于openmp自带的+,这里命名为merge
typename-list: 归约操作的数据类型，这里为std::vector <int>
combiner: 合并链接具体操作，insert为具体操作，omp_out与omp_in为固定的标识符
initializer-clause:
归约操作的每个线程的初始值，比如求和操作时赋值100则等效于100xn
 */

  std::vector<int> vec;
#pragma omp parallel for reduction(merge : vec)
  for (int i = 0; i < 10; i++)
    vec.push_back(i);

  // // 乱序
  // #pragma omp parallel
  //   {
  //     std::vector<int> vec_private;
  // #pragma omp for nowait // fill vec_private in parallel
  //     for (int i = 0; i < 10; i++) {
  //       vec_private.push_back(i);
  //     }
  // #pragma omp critical
  //     vec.insert(vec.end(), vec_private.begin(), vec_private.end());
  //   }

  // // 顺序
  // #pragma omp parallel
  //   {
  //     std::vector<int> vec_private;
  // #pragma omp for nowait schedule(static)
  //     for (int i = 0; i < 10; i++) {
  //       vec_private.push_back(i);
  //     }
  // #pragma omp for schedule(static) ordered
  //     for (int i = 0; i < omp_get_num_threads(); i++) {
  // #pragma omp ordered
  //       vec.insert(vec.end(), vec_private.begin(), vec_private.end());
  //     }
  //   }

  // // 顺序
  //   std::vector<int> vec;
  //   size_t *prefix;
  // #pragma omp parallel
  //   {
  //     int ithread = omp_get_thread_num();
  //     int nthreads = omp_get_num_threads();
  // #pragma omp single
  //     {
  //       prefix = new size_t[nthreads + 1];
  //       prefix[0] = 0;
  //     }
  //     std::vector<int> vec_private;
  // #pragma omp for schedule(static) nowait
  //     for (int i = 0; i < 10; i++) {
  //       vec_private.push_back(i);
  //     }
  //     prefix[ithread + 1] = vec_private.size();
  // #pragma omp barrier
  // #pragma omp single
  //     {
  //       for (int i = 1; i < (nthreads + 1); i++)
  //         prefix[i] += prefix[i - 1];
  //       vec.resize(vec.size() + prefix[nthreads]);
  //     }
  //     std::copy(vec_private.begin(), vec_private.end(),
  //               vec.begin() + prefix[ithread]);
  //   }
  //   delete[] prefix;

  for (const auto &val : vec)
    std::cout << val << std::endl;
  return 0;
}