#include <cassert>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <thrust/device_vector.h>

// row-major
template <typename T>
__global__ void fillMatrix(const int rows, const int cols, T *d_outMatrix) {
  const size_t tx = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t ty = threadIdx.y + blockIdx.y * blockDim.y;

  if (tx < cols && ty < rows) {
    const size_t idx = ty * cols + tx;
    d_outMatrix[idx] = (T)idx;
  }
}

template <typename T>
struct linear_index_to_row_index : public thrust::unary_function<T, T> {
  T C; // number of columns

  __host__ __device__ linear_index_to_row_index(T C) : C(C) {}

  __host__ __device__ T operator()(T i) { return i / C; }
};

template <typename T>
void launch_ThrustRowSumReduce(
    const int &rows, const int &columns,
    const thrust::device_vector<T> &d_matrix,
    thrust::device_vector<T> &row_sums,
    const cudaStream_t &stream = nullptr) // thrust::universal_vector
{
  assert(row_sums.size() == rows);

  thrust::device_vector<int> row_indices(rows);

  if (stream)
    thrust::reduce_by_key(
        thrust::cuda::par.on(stream),
        thrust::make_transform_iterator(
            thrust::counting_iterator<int>(0),
            linear_index_to_row_index<int>(columns)),
        thrust::make_transform_iterator(thrust::counting_iterator<int>(0),
                                        linear_index_to_row_index<int>(rows)) +
            (rows * columns),
        d_matrix.begin(), row_indices.begin(), row_sums.begin(),
        thrust::equal_to<int>(), thrust::plus<T>());
  else
    thrust::reduce_by_key(thrust::make_transform_iterator(
                              thrust::counting_iterator<int>(0),
                              linear_index_to_row_index<int>(columns)),
                          thrust::make_transform_iterator(
                              thrust::counting_iterator<int>(0),
                              linear_index_to_row_index<int>(columns)) +
                              (rows * columns),
                          d_matrix.begin(), row_indices.begin(),
                          row_sums.begin(), thrust::equal_to<int>(),
                          thrust::plus<T>());
}

template <typename T>
void printMatrix(const int &rows, const int &cols,
                 const thrust::device_vector<T> &d_vec) {
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      if (j != cols - 1)
        std::cout << d_vec[i * cols + j] << ", ";
      else
        std::cout << d_vec[i * cols + j] << std::endl;
    }
  }
}

int main() {
  const int rows = 2, cols = 4;
  dim3 blockSize(16, 64, 1);
  dim3 gridSize((cols + blockSize.x - 1) / blockSize.x,
                (rows + blockSize.y - 1) / blockSize.y, 1);

  thrust::device_vector<double> d_vec(rows * cols);
  fillMatrix<double><<<gridSize, blockSize>>>(rows, cols, d_vec.data().get());
  cudaDeviceSynchronize();

  printMatrix(rows, cols, d_vec);

  // 测试thrust求和
  thrust::device_vector<double> d_rowSum(rows);
  launch_ThrustRowSumReduce<double>(rows, cols, d_vec, d_rowSum);
  cudaDeviceSynchronize();

  printMatrix(rows, 1, d_rowSum);
  return 0;
}