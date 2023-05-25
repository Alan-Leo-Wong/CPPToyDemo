#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <igl/point_mesh_squared_distance.h>
#include <igl/read_triangle_mesh.h>
#include <igl/writeOBJ.h>
#include <igl/writeOFF.h>
#include <iostream>
#include <random>
#include <string>

int main() {
  // 读取网格模型
  Eigen::MatrixXd V;
  Eigen::MatrixXi F;
  igl::read_triangle_mesh("bunny.obj", V, F);

  // 设置扰动噪声的百分比
  double noisePercentage = 0.0025; // 0.25%

  // 计算需要扰动的节点数量
  int numNodes = V.rows();
  int numNoisyNodes = static_cast<int>(noisePercentage * numNodes);

  // 设置随机数生成器
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dis(
      -0.008,
      0.008); // 设置扰动范围为 [-0.1, 0.1]

  // 随机扰动节点位置
  for (int i = 0; i < numNoisyNodes; i++) {
    // 随机选择一个节点索引
    int nodeIndex = std::rand() % numNodes;

    // 在节点位置上添加随机扰动
    V.row(nodeIndex) += dis(gen) * Eigen::RowVector3d::Random();
  }

  // 保存带有噪声的网格模型
  std::string out_file = "noisy_bunny.obj";
  igl::writeOBJ(out_file, V, F);

  std::cout << "Noise added to the mesh and saved as " << out_file << std::endl;

  return 0;
}
