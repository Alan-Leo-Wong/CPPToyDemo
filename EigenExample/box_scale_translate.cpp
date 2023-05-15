#include <Eigen/Dense>
#include <iostream>

class Box {
public:
  Box(const Eigen::Vector3f &min_point, const Eigen::Vector3f &max_point)
      : min_point_(min_point), max_point_(max_point) {}

  void ScaleAndTranslate(const float scale_factor,
                         const Eigen::Vector3f &translation) {
    // 计算中心点
    const Eigen::Vector3f center = (min_point_ + max_point_) / 2.0;

    // 以中心点为基准进行缩放和平移
    const Eigen::Vector3f scaled_min_point =
        (min_point_ - center) * scale_factor + center + translation;
    const Eigen::Vector3f scaled_max_point =
        (max_point_ - center) * scale_factor + center + translation;

    // 更新边界框的坐标
    min_point_ = scaled_min_point;
    max_point_ = scaled_max_point;
  }

  void Print() const {
    std::cout << "min_point: " << min_point_.transpose() << std::endl;
    std::cout << "max_point: " << max_point_.transpose() << std::endl;
  }

private:
  Eigen::Vector3f min_point_;
  Eigen::Vector3f max_point_;
};

int main() {
  // 创建一个边界框
  Eigen::Vector3f min_point(0.0, 0.0, 0.0);
  Eigen::Vector3f max_point(1.0, 1.0, 1.0);
  Box box(min_point, max_point);

  // 缩放因子和平移向量
  const float scale_factor = 2.0;
  Eigen::Vector3f translation(1.0, 2.0, 3.0);

  // 对边界框进行缩放和平移
  box.ScaleAndTranslate(scale_factor, translation);

  // 打印更新后的边界框坐标
  box.Print();

  return 0;
}
