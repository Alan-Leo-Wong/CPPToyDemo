/*
 * @Author: WangLei
 * @authorEmail: leiw1006@gmail.com
 * @Date: 2023-05-15 21:34:39
 * @LastEditors: WangLei
 * @LastEditTime: 2023-05-15 21:46:28
 * @FilePath: \3DPointSampleGeneraor\gaussian_sample.cpp
 * @Description: 
 */
#include <fstream>
#include <iostream>
#include <random>
#include <Eigen/Dense>

// 高斯采样函数
Eigen::Vector3d gaussianSample(const Eigen::Vector3d& mean, const Eigen::Vector3d& stddev) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, 1.0);

    Eigen::Vector3d sample;
    for (int i = 0; i < 3; ++i) {
        sample(i) = mean(i) + stddev(i) * dist(gen);
    }

    return sample;
}

// 高斯采样三维空间函数
void gaussianSample3D(const Eigen::Vector3d& min_area, const Eigen::Vector3d& max_area, int num_samples) {
    Eigen::Vector3d mean = (max_area + min_area) / 2.0;
    Eigen::Vector3d stddev = (max_area - min_area) / 6.0;

    std::ofstream out(".\\gaussian_points.xyz");
    for (int i = 0; i < num_samples; ++i) {
        // 生成高斯样本
        Eigen::Vector3d sample = gaussianSample(mean, stddev);

        // 打印样本
        // std::cout << "Sample " << i + 1 << ": (" << sample(0) << ", " << sample(1) << ", " << sample(2) << ")" << std::endl;
        out << sample(0) << " " << sample(1) << " " << sample(2) << std::endl;
    }
}

int main() {
    Eigen::Vector3d min_area(0.0, 0.0, 0.0);    // 最小边界
    Eigen::Vector3d max_area(10.0, 10.0, 10.0); // 最大边界
    int num_samples = 100000;                        // 采样数量

    // 进行高斯采样
    gaussianSample3D(min_area, max_area, num_samples);

    return 0;
}
