/*
 * @Author: Alan Wang leiw1006@gmail.com
 * @Date: 2023-12-13 19:14:42
 * @LastEditors: Alan Wang leiw1006@gmail.com
 * @LastEditTime: 2023-12-13 23:38:00
 * @FilePath: \EigenExample\sparse_matrix.cpp
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置
 * 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#include <Eigen/Sparse>
#include <cstdlib>
#include <iostream>
#include <unordered_set>
#include <unsupported/Eigen/SparseExtra>
#include <vector>

void traverseSparseMatrix(const Eigen::SparseMatrix<double> &G) {
  // // 默认列优先，所以默认按列遍历
  // std::cout << "colum major traverse:\n";
  // for (int k = 0; k < G.outerSize(); ++k) {
  //   for (Eigen::SparseMatrix<double>::InnerIterator it(G, k); it; ++it) {
  //     std::cout << "row: " << it.row() << ", col: " << it.col()
  //               << ", value: " << it.value() << std::endl;
  //   }
  // }

  // std::cout << "=============\n";

  // std::cout << "row major traverse:\n";
  // 使用行迭代器遍历该行的非零元素
  for (int i = 0; i < G.outerSize(); ++i) {
    for (int k = G.outerIndexPtr()[i]; k < G.outerIndexPtr()[i + 1]; ++k) {
      int colIndex = G.innerIndexPtr()[k];
      double value = G.valuePtr()[k];

      // 输出列索引和对应的值
      std::cout << "row: " << i << ", col: " << colIndex << ", value: " << value
                << std::endl;
    }
  }
}

// 输出无向图G与源节点v(准确来说是第v行所代表的节点)相邻的节点
void printAdjToV(const Eigen::SparseMatrix<double> &G, int v) {
  std::cout << "v: " << v << std::endl;
  // // 使用列优先遍历v这一列
  // for (Eigen::SparseMatrix<double>::InnerIterator it(G, v); it; ++it) {
  //   std::cout << "adj: " << it.row() << ", value: " << it.value() <<
  //   std::endl;
  // }
  // 也可以使用行优先遍历v这一行
  for (int k = G.outerIndexPtr()[v]; k < G.outerIndexPtr()[v + 1]; ++k) {
    int colIndex = G.innerIndexPtr()[k];
    double value = G.valuePtr()[k];
    std::cout << "adj: " << colIndex << ", value: " << value << std::endl;
  }
}

// 获取无向图G的节点v(准确来说是第v行所代表的节点)的诱导子图
Eigen::SparseMatrix<double>
getInducedSubgraph(const Eigen::SparseMatrix<double> &G, int v) {
  std::vector<Eigen::Triplet<double>> subGTriplet;
  subGTriplet.reserve(G.rows());
  // 图 G 中节点索引到图 subG 节点索引的映射
  std::unordered_map<int, int> GToSubG;
  int subGIdx = 0;
  GToSubG[v] = subGIdx++;
  std::unordered_set<int> neighborSet;

  // 得到节点v的度数
  int degree = 0;
  // 遍历节点v的邻居节点
  for (int k = G.outerIndexPtr()[v]; k < G.outerIndexPtr()[v + 1]; ++k) {
    ++degree;
    int neighbor = G.innerIndexPtr()[k];

    GToSubG[neighbor] = subGIdx;
    double value = G.valuePtr()[k];
    neighborSet.insert(neighbor);

    // 添加边(v, neighbor)和(neighbor, v)到诱导子图
    subGTriplet.emplace_back(0, subGIdx, value);
    subGTriplet.emplace_back(subGIdx, 0, value);
    ++subGIdx;
  }
  // v的诱导子图等于 节点v的度数 + 1
  int numNodes = degree + 1;

  // 添加与节点v相邻的节点之间的边
  for (int k = G.outerIndexPtr()[v]; k < G.outerIndexPtr()[v + 1]; ++k) {
    int neighbor = G.innerIndexPtr()[k];

    for (const int node : neighborSet) {
      if (G.coeff(neighbor, node) != .0) {
        subGTriplet.emplace_back(GToSubG.at(neighbor), GToSubG.at(node),
                                 G.coeff(neighbor, node));
      }
    }

    // for (int n_k = G.outerIndexPtr()[neighbor];
    //      n_k < G.outerIndexPtr()[neighbor + 1]; ++n_k) {
    //   int neighborNeighbor = G.innerIndexPtr()[n_k];
    //   double value = G.valuePtr()[n_k];
    //   if (neighborSet.count(neighborNeighbor)) {
    //     subGTriplet.emplace_back(neighbor, neighborNeighbor, value);
    //     subGTriplet.emplace_back(neighborNeighbor, neighbor, value);
    //   }
    // }
  }

  // 创建一个稀疏矩阵表示诱导子图
  Eigen::SparseMatrix<double> subG(numNodes, numNodes);
  subG.setFromTriplets(subGTriplet.begin(), subGTriplet.end());

  return subG;
}

int main() {
  Eigen::SparseMatrix<double> G(4, 4);
  G.insert(0, 1) = 2.0;
  G.insert(1, 0) = 2.0;

  G.insert(1, 2) = 3.0;
  G.insert(2, 1) = 3.0;
  G.insert(1, 3) = 4.0;
  G.insert(3, 1) = 4.0;

  G.insert(2, 3) = 5.0;
  G.insert(3, 2) = 5.0;
  G.makeCompressed();

  // traverseSparseMatrix(G);
  // std::cout << "=============\n";

  printAdjToV(G, 2);
  std::cout << "=============\n";

  Eigen::saveMarket(G, R"(D:\CPPToyDemo\EigenExample\graph\graph.mtx)");

  Eigen::SparseMatrix<double> subG = getInducedSubgraph(G, 2);
  traverseSparseMatrix(subG);

  Eigen::saveMarket(subG, R"(D:\CPPToyDemo\EigenExample\graph\sub_graph.mtx)");

  return 0;
}