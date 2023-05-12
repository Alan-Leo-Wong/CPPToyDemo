#include <Eigen\Dense>
#include <iostream>
#include <vector>

using namespace std;

constexpr int rows = 2;

template <typename T> class PreAllocator {
private:
  T *memory_ptr;
  std::size_t memory_size;

public:
  typedef std::size_t size_type;
  typedef T *pointer;
  typedef T value_type;

  PreAllocator(T *memory_ptr, std::size_t memory_size)
      : memory_ptr(memory_ptr), memory_size(memory_size) {}

  PreAllocator(const PreAllocator &other) throw()
      : memory_ptr(other.memory_ptr), memory_size(other.memory_size){};

  template <typename U>
  PreAllocator(const PreAllocator<U> &other) throw()
      : memory_ptr(other.memory_ptr), memory_size(other.memory_size){};

  template <typename U> PreAllocator &operator=(const PreAllocator<U> &other) {
    return *this;
  }
  PreAllocator<T> &operator=(const PreAllocator &other) { return *this; }
  ~PreAllocator() {}

  pointer allocate(size_type n, const void *hint = 0) { return memory_ptr; }
  void deallocate(T *ptr, size_type n) {}

  size_type max_size() const { return memory_size; }
};

int main() {
  Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> mat(rows, 3);
  for (int i = 0; i < rows; ++i)
    mat.row(i) = Eigen::Vector3d(i - 1, i, i + 1);

  Eigen::Vector3d *buf = reinterpret_cast<Eigen::Vector3d *>(mat.data());
  std::vector<Eigen::Vector3d, PreAllocator<Eigen::Vector3d>> vec_1(
      rows, PreAllocator<Eigen::Vector3d>(buf, rows));

  for (int i = 0; i < rows; ++i)
    std::cout << vec_1[i].transpose() << std::endl;

  std::cout << "=========\n";

  std::vector<Eigen::Vector3d> vec_2(rows);
  Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>::Map(
      vec_2.data()->data(), rows, 3) = mat;

  for (int i = 0; i < rows; ++i)
    std::cout << vec_2[i].transpose() << std::endl;

  //   std::cout << "=========\n";

  //   Eigen::Matrix<double, 3, Eigen::Dynamic> c_mat(rows, 3);
  //   std::vector<Eigen::Vector3d> vec_3(rows);
  //   Eigen::Matrix<double, 3, Eigen::Dynamic>::Map(vec_3.data()->data(), 3,
  //   rows) =
  //       c_mat;

  //   for (int i = 0; i < rows; ++i)
  //     std::cout << vec_3[i].transpose() << std::endl;
  return 0;
}
