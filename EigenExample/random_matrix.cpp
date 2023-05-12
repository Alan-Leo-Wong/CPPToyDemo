#include <Eigen/Dense>
#include <algorithm>
#include <ctime>
#include <iostream>
#include <numeric>
#include <random>

template <typename Scalar>
inline bool getRandomMatrix(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &area,
    const size_t &num, const float &scaleFactor, const float &dis,
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &M) {
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> _M(num, 3);
  std::default_random_engine e(time(0)); // current time as seed
  std::normal_distribution<Scalar> n(0, 1);
  _M = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Zero(num, 3)
           .unaryExpr([&](Scalar val) {
             return static_cast<Scalar>(round((n(e) + 1e-6) * 1e6) / 1e6);
           });

  M.resize(num, 4);
  M.setOnes();
  M.block(0, 0, num, 3) = _M.block(0, 0, num, 3);

  const auto min_area = area.colwise().minCoeff();
  const auto max_area = area.colwise().maxCoeff();
  const auto min_m = M.colwise().minCoeff();
  const auto max_m = M.colwise().maxCoeff();

  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> diag_area =
      max_area - min_area; // bounding box
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> diag_m =
      max_m - min_m; // random points

  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> center_area =
      0.5 * (max_area + min_area);
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> center_m = 0.5 * (max_m + min_m);

  Eigen::Matrix<Scalar, 4, 4> zoomMatrix =
      Eigen::Matrix<Scalar, 4, 4>::Identity();
  Eigen::Matrix<Scalar, 4, 4> transMatrix =
      Eigen::Matrix<Scalar, 4, 4>::Identity();

  for (int d = 0; d < 3; d++) {
    if (diag_m[d] == 0)
      return false;
    zoomMatrix(d, d) = (diag_area[d] / diag_m[d]) * scaleFactor;
    transMatrix(3, d) = (center_area[d] - center_m[d] * zoomMatrix(d, d)) + dis;
  }

  M = M * zoomMatrix * transMatrix;
  M = M.block(0, 0, num, 3);
  M = (M.array() < 1e-6).select(0, M);

  return true;
}

template <typename Scalar>
inline bool
getRandomMatrix(const Eigen::Matrix<Scalar, 3, 1> &min_area,
                const Eigen::Matrix<Scalar, 3, 1> &max_area, const size_t &num,
                const float &scaleFactor, const float &dis,
                Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &M) {
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> _M(num, 3);
  std::default_random_engine e(time(0)); // current time as seed
  std::normal_distribution<Scalar> n(-1, 1);
  // std::uniform_real_distribution<Scalar> n(-1, 1);
  _M = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Zero(num, 3)
           .unaryExpr([&](Scalar val) {
             return static_cast<Scalar>(round((n(e) + 1e-6) * 1e6) / 1e6);
           });

  _M.conservativeResize(num + 1, 4); // strange value because reallocate memory
  Eigen::Matrix<Scalar, 1, 4> r_zeroVec(num);
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> c_onesVec(num + 1);
  r_zeroVec.setZero();
  c_onesVec.setOnes();
  _M.row(num) = r_zeroVec;
  _M.col(3) = c_onesVec;

  // Eigen::Matrix<Scalar, Eigen::Dynamic, 1> zeroVec(num);
  // zeroVec.setZero();
  // _M.row(num) = zeroVec;
  // _M.col(3) = zeroVec;
  // _M(num, 3) = 1;
  // std::cout << _M.col(3) << std::endl;
  // std::cout << _M.row(num) << std::endl;

  // M.resize(num, 4);
  // M.setOnes();
  // M.block(0, 0, num, 3) = _M.block(0, 0, num, 3);

  const auto min_m = _M.colwise().minCoeff();
  const auto max_m = _M.colwise().maxCoeff();

  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> diag_area =
      max_area - min_area; // bounding box
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> diag_m =
      max_m - min_m; // random points

  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> center_area =
      0.5 * (max_area + min_area);
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> center_m = 0.5 * (max_m + min_m);

  Eigen::Matrix<Scalar, 4, 4> zoomMatrix =
      Eigen::Matrix<Scalar, 4, 4>::Identity();
  Eigen::Matrix<Scalar, 4, 4> transMatrix =
      Eigen::Matrix<Scalar, 4, 4>::Identity();

  for (int d = 0; d < 3; d++) {
    if (diag_m[d] == 0)
      return false;
    zoomMatrix(d, d) = (diag_area[d] / diag_m[d]) * scaleFactor;
    transMatrix(3, d) = (center_area[d] - center_m[d] * zoomMatrix(d, d)) + dis;
  }

  _M = _M * zoomMatrix * transMatrix;
  M.resize(num, 3);
  M = _M.block(0, 0, num, 3);
  M = (M.array().abs() < 1e-9).select(0, M);

  return true;
}

int main() {
  Eigen::MatrixXf area(2, 3);
  Eigen::MatrixXf M;
  area.row(0) = Eigen::Vector3f(-1000, -1000, -1000);
  area.row(1) = Eigen::Vector3f(1000, 1000, 1000);
  //   getRandomMatrix<1, 3>(area, 0.5, 0.5, M);
  const Eigen::Vector3f min_area = area.colwise().minCoeff();
  const Eigen::Vector3f max_area = area.colwise().maxCoeff();
  if (getRandomMatrix<float>(min_area, max_area, 1000000, 1.0f, 0.0f, M)) {
    // std::cout << "=====\n";
    for (int i = 0; i < M.rows(); ++i)
      std::cout << M.row(i) << std::endl;
  }

  return 0;
}