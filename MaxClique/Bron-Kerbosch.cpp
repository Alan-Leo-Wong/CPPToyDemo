/*
 * @Author: WangLei
 * @authorEmail: leiw1006@gmail.com
 * @Date: 2023-11-29 14:49:20
 * @LastEditors: Poet 602289550@qq.com
 * @LastEditTime: 2023-11-29 21:51:00
 * @FilePath: \MaxClique\Bron-Kerbosch.cpp
 * @Description:
 */
#include <cstdio>
#include <cstring>
#include <iostream>
#include <iterator>
#include <list>
#include <system_error>
#include <vector>
using namespace std;

/**
 * @brief ref:
 * https://bowiee.github.io/2019/08/07/Bron-Kerbosch%E7%AE%97%E6%B3%95/
 *
 */

const int maxn = 130;

int node[maxn]; // 存储节点值(** 假设节点值都 "大于等于1" **)
bool mp[maxn][maxn]; // 表示结点之间的连接关系，1代表连接

int some[maxn][maxn]; // 集合 P (备选集合)
int none[maxn][maxn]; // 集合 X (已选集合, 在回溯时由集合 R 中拿出, 因此集合 X
                      // 中的顶点一定可以和 R 集合中的顶点构成一个团,
                      // 所以必须保证集合 P 和 X 都为空时, 此时的 R
                      // 才是一个极大团)
                      // 在递归到下一层时集合 X 也要取与当前从集合 P
                      // 拿出的备选顶点 v 的邻居集合的交集)
int all[maxn][maxn]; // 集合 R (极大团集合)
// 以上三个集合的第一个维度代表 dfs 的深度, 第二个维度代表结点值

int n, m, ans; // n 表示结点数, m 表示边数, ans 表示极大团数量

/**
 * @brief 普通版本的 Bron-Kerbosch Algorithm
 *
 * @param d 搜索深度
 * @param an all(R)集合中顶点数
 * @param sn some(P)集合中顶点数
 * @param nn none(X)集合中顶点数
 */
void trivalDFS(int d, int an, int sn, int nn) {
  // sn==0 and nn==0 时(集合 P 和集合 X 均为空)
  if (sn == 0 && nn == 0) {
    ++ans;
    // 输出集合
    printf("max clique: ");
    for (int j = 0; j < an; ++j)
      printf("%d ", all[d][j]);
    printf("\n");
  }
  if (sn == 0)
    return;

  for (int i = 0; i < sn; ++i) { // 遍历 P 中的结点
    int v = some[d][i]; // 取出 P 中的第 i 个结点作为备选节点

    // 将当前层的 R 集合中的节点 加上 v , 作为下一层的集合 R
    for (int j = 0; j < an; ++j)
      all[d + 1][j] = all[d][j];
    all[d + 1][an] = v;

    // 用来分别记录下一层中 P 和 X 中的结点个数
    int next_sn = 0, next_nn = 0;

    // 更新输入到下一层的集合 P
    for (int j = 0; j < sn; ++j)
      if (mp[v][some[d][j]])
        some[d + 1][next_sn++] =
            some[d][j]; // 从当前层的集合 P 中挑选出与 v 相连接的节点,
                        // 作为下一层的集合 P (保证了 P 与现有集合 R
                        // 中所有的点相连接)

    // 更新输入到下一层的集合 X
    for (int j = 0; j < nn; ++j)
      if (mp[v][none[d][j]])
        none[d + 1][next_nn++] =
            none[d][j]; // 从当前层的集合 X 中挑选出与 v 相连接的节点,
                        // 作为下一层的集合 X (保证了 X 与现有集合 R
                        // 中所有的点相连接)

    trivalDFS(d + 1, an + 1, next_sn, next_nn);

    // 回溯, 更新集合 P 和集合 X
    some[d][i] = 0; // 节点值为0的节点不存在我们的图中,
                    // 即邻接矩阵mp中不会有关于0这个节点的邻接信息
    none[d][nn++] = v; // 将已经操作过的节点 v 放入集合 X 中
  }
}

/**
 * @brief 带 pivot 剪枝的 Bron-Kerbosch Algorithm
 *
 * @param d 搜索深度
 * @param an all(R)集合中顶点数
 * @param sn some(P)集合中顶点数
 * @param nn none(X)集合中顶点数
 */
void pivotDFS(int d, int an, int sn, int nn) {
  // sn==0 and nn==0 时(集合 P 和集合 X 均为空)
  if (sn == 0 && nn == 0) {
    ++ans;
    // 输出集合
    printf("max clique: ");
    for (int j = 0; j < an; ++j)
      printf("%d ", all[d][j]);
    printf("\n");
  }
  if (sn == 0)
    return;

  // 选取 Pivot 结点(Pivot 结点的邻居结点数量越多, 递归次数越少)
  int u = some[d][0];

  for (int i = 0; i < sn; ++i) { // 遍历 P 中的结点
    int v = some[d][i]; // 取出 P 中的第 i 个结点作为备选节点

    // 如果是邻居结点, 就直接跳过下面的程序, 进行下一轮的循环。
    // 显然能让程序运行下去的, 只有两种, 一种是 v 就是 u 结点本身,
    // 另一种则是 v 不是 u 的邻居结点
    // 从 u 的邻居节点找下去一定能找到包括 u 的极大团(即通过后面 P 和 X 的更新),
    // 所以没必要再从 u 的邻居作为"首发"往下找了
    if (mp[u][v])
      continue;
    // printf("u = %d, v = %d\n", u, v);
    // system("pause");

    // 将当前层的 R 集合中的节点 加上 v , 作为下一层的集合 R
    for (int j = 0; j < an; ++j)
      all[d + 1][j] = all[d][j];
    all[d + 1][an] = v;

    // 用来分别记录下一层中 P 和 X 中的结点个数
    int next_sn = 0, next_nn = 0;

    // 更新输入到下一层的集合 P
    for (int j = 0; j < sn; ++j)
      if (mp[v][some[d][j]])
        some[d + 1][next_sn++] =
            some[d][j]; // 从当前层的集合 P 中挑选出与 v 相连接的节点,
                        // 作为下一层的集合 P (保证了 P 与现有集合 R
                        // 中所有的点相连接)

    // 更新输入到下一层的集合 X
    for (int j = 0; j < nn; ++j)
      if (mp[v][none[d][j]])
        none[d + 1][next_nn++] =
            none[d][j]; // 从当前层的集合 X 中挑选出与 v 相连接的节点,
                        // 作为下一层的集合 X (保证了 X 与现有集合 R
                        // 中所有的点相连接)

    pivotDFS(d + 1, an + 1, next_sn, next_nn);

    // 回溯, 更新集合 P 和集合 X
    some[d][i] = 0; // 节点值为0的节点不存在我们的图中,
                    // 即邻接矩阵mp中不会有关于0这个节点的邻接信息
    none[d][nn++] = v; // 将已经操作过的节点 v 放入集合 X 中
  }
}

template <typename T>
void BronKerbosch(std::list<T> &R, std::list<T> &P, std::list<T> &X,
                  std::vector<std::list<T>> &all_max_clique) {
  if (P.empty() && X.empty()) {
    all_max_clique.emplace_back(R);
  }
  if (P.empty())
    return;

  auto pivot_iter = P.begin();
  int pivot = *pivot_iter;
  while (pivot == -1 && ++pivot_iter != P.end()) {
    pivot = *pivot_iter;
  }
  if (pivot_iter == P.end()) {
    std::cerr << "Error! NO VALID PIVOT IN 'computeMaxClique'!";
    exit(1);
  }
  for (auto &p_1 : P) {
    if (p_1 == -1)
      continue;
    if (mp[pivot][p_1])
      continue;

    std::list<T> next_R = R;
    next_R.emplace_back(p_1);

    std::list<T> next_P;
    for (const auto &p_2 : P)
      if (p_2 != -1 && mp[p_1][p_2])
        next_P.emplace_back(p_2);

    std::list<T> next_X;
    for (const auto &x : X)
      if (x != -1 && mp[p_1][x])
        next_X.emplace_back(x);

    BronKerbosch(next_R, next_P, next_X, all_max_clique);

    // iter = P.erase(iter);
    X.emplace_back(p_1);
    p_1 = -1;
  }
}

void work() {
  ans = 0;

  // 初始化 P 集合, 保存全部节点(可以将以下循环放至main函数对node的输入处)
  for (int i = 0; i < n; ++i)
    some[0][i] = node[i];

  // 初始时 R 集合和 X 集合均为空
  // trivalDFS(0, 0, n, 0);
  pivotDFS(0, 0, n, 0);
}

template <typename T> void exec(std::vector<std::list<T>> &all_max_clique) {
  std::list<T> R;
  std::list<T> P;
  std::list<T> X;

  for (int i = 0; i < n; ++i)
    P.emplace_back(node[i]);

  BronKerbosch(R, P, X, all_max_clique);
}

/**
4 4
1 2 3 4
1 2
1 3
2 3
2 4

7 10
1 2 3 4 5 6 7
1 2
1 3
1 4
2 3
2 4
2 5
3 4
4 5
4 6
5 7
 */

int main() {
  while (~scanf_s("%d %d", &n, &m)) {
    memset(node, 0, sizeof node);
    for (int i = 0; i < n; ++i) {
      int v;
      scanf_s("%d", &v);
      node[i] = v;
    }

    memset(mp, 0, sizeof mp);
    for (int i = 0; i < m; ++i) {
      int u, v;
      scanf_s("%d %d", &u, &v);
      mp[u][v] = mp[v][u] = 1;
    }

    // work();
    std::vector<std::list<int>> all_max_clique;
    exec<int>(all_max_clique);

    // printf("The number of max cliques = %d\n", ans);
    printf("The number of max cliques = %lld\n", all_max_clique.size());
    for (const auto &max_clique : all_max_clique) {
      std::cout << "max clique: ";
      for (const auto &val : max_clique) {
        std::cout << val << " ";
      }
      std::cout << "\n";
    }
  }
  return 0;
}