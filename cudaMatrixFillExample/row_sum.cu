#include "device_query.cuh"
#include <cassert>
#include <chrono>
#include <cooperative_groups.h>
#include <cublas_v2.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/pair.h>
#include <thrust/transform_reduce.h>
#include <thrust/unique.h>
#include <vector>

namespace cg = cooperative_groups;

inline bool isPow2(const unsigned int &x) { return ((x & (x - 1)) == 0); }

inline unsigned int nextPow2(unsigned int x) {
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return ++x;
}

// 确保列数是warpSize的整数倍，以便正确地进行行求和
#define PADDING(nvRows)                                                        \
  ((nvRows % 32 == 0) ? (nvRows) : (nvRows + 32 - (nvRows % 32)))

#ifndef MIN
#define MIN(x, y) ((x < y) ? x : y)
#endif // !MIN

#ifndef MAX
#define MAX(x, y) ((x > y) ? x : y)
#endif // !MAX

inline void getBlocksAndThreadsNum(const cudaDeviceProp &prop,
                                   const unsigned int &n, const int &maxBlocks,
                                   const int &maxThreads, int &blocks,
                                   int &threads) {
  // 最多使用n/2的线程数
  threads = (n < maxThreads * 2) ? nextPow2((n + 1) / 2) : maxThreads;
  threads = isPow2(threads) ? nextPow2(threads) : threads;

  // 加入maxBlocks可帮助实现网格级别的跨度
  // (n + (threads * 2) - 1) / (threads * 2)是为了实现一个线程块级别的跨度
  blocks = MIN(maxBlocks, (n + (threads * 2) - 1) / (threads * 2));

  if ((float)threads * blocks >
      (float)prop.maxGridSize[0] * prop.maxThreadsPerBlock) {
    printf("n is too large, please choose a smaller number!\n");
    exit(EXIT_FAILURE);
  }

  if (blocks > prop.maxGridSize[0]) {
    printf("Grid size <%d> exceeds the device capability <%d>\n", blocks,
           prop.maxGridSize[0]);
    const int t_blocks = blocks;
    const int t_threads = threads;
    while (blocks > prop.maxGridSize[0]) {
      if (threads * 2 <= maxThreads) {
        threads *= 2;
        blocks = (n + (threads * 2) - 1) / (threads * 2);
      } else {
        break;
      }
    }
    printf("Set grid size as <%d> (original is <%d>), set block size as <%d> "
           "(original is "
           "<%d>)\n",
           blocks, t_blocks, threads, t_threads);
  }
  // printf("-- Final grid size = %d, block size = %d\n", blocks, threads);
}

template <typename T> struct SharedMemory {
  __device__ inline operator T *() {
    extern __shared__ int __smem[];
    return (T *)__smem;
  }

  __device__ inline operator const T *() const {
    extern __shared__ int __smem[];
    return (T *)__smem;
  }
};

// sum reduce at warp level
template <typename T>
__device__ __forceinline__ void warpReduceSum(unsigned int mask, T &sum) {
  for (int offset = warpSize >> 1; offset > 0; offset >>= 1)
    sum += __shfl_down_sync(mask, sum, offset);
}

template <typename T>
__device__ __forceinline__ T func(const T &ai, const T &bj) {
  return ai * bj;
}

/*
 * matrix reduce for sum of row
 * @param m: rows
 * @param n: columns, and n % warpSize = 0
 * @param g_idata: matrix(m * n)
 */
template <typename T = double, bool nIsPow2, unsigned int colBlockSize>
__global__ void reduceRowSumKernel(const unsigned int m, const unsigned int n,
                                   const T *__restrict__ g_iA,
                                   const T *__restrict__ g_iB,
                                   T *__restrict__ g_odata) {
  T *shData = SharedMemory<T>();
  cg::thread_block ctb = cg::this_thread_block();

  unsigned int ty = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int tx = blockIdx.x * blockDim.x + threadIdx.x;

  if (ty < m) {
    unsigned int x_tid = threadIdx.x;
    unsigned int x_gridSize = colBlockSize * gridDim.x;

    unsigned int maskLength = (colBlockSize & 31);
    maskLength = (maskLength > 0) ? (32 - maskLength) : maskLength;
    const unsigned int mask = (0xffffffff) >> maskLength;

    T sum = 0;

    // reduce multiple elements per thread
    if (nIsPow2) {
      unsigned int i = blockIdx.x * colBlockSize * 2 + threadIdx.x;
      x_gridSize <<= 1;

      while (i < n) {
        sum += func<T>(g_iA[ty], g_iB[i]);

        if (i + colBlockSize < n)
          sum += func<T>(g_iA[ty],
                         g_iB[i + colBlockSize]); // (一个)线程块级别的跨度
        i +=
            x_gridSize; // 网格级别的跨度：默认网格大小(block的数量)为原有数据(x维度即列数)的一半
      }
    } else {
      unsigned int i = blockIdx.x * colBlockSize + threadIdx.x;
      while (i < n) {
        sum += func<T>(g_iA[ty], g_iB[i]);
        i += x_gridSize;
      }
    }

    // 对每个warp执行归约求和，然后保存到shared memory中
    warpReduceSum<T>(mask, sum);
    const int sh_reduceNum =
        (colBlockSize / warpSize) > 0 ? colBlockSize / warpSize : 1;
    if (x_tid % warpSize == 0)
      shData[threadIdx.y * sh_reduceNum + x_tid / warpSize] = sum;

    cg::sync(ctb);

    // 同一个block下所有warp求和(只要将每个warp的第一个thread保存的sum加起来即可，
    // 因为每个warp的第一个thread保存的sum就是其所属warp的所有线程的数据和)
    const unsigned int newMask = __ballot_sync(mask, x_tid < sh_reduceNum);
    if (x_tid < sh_reduceNum) {
      sum = shData[threadIdx.y * sh_reduceNum + x_tid];
      warpReduceSum<T>(newMask, sum);
    }

    if (x_tid == 0) {
      g_odata[ty * gridDim.x + blockIdx.x] = sum;
    }
  }
}

template <typename T>
void switchKernel(const bool &isPow2, const int &threads, const dim3 &gridSize,
                  const dim3 &blockSize, const int &sh_memSize,
                  const cudaStream_t &stream, const int &rowElems,
                  const int &paddingCols, T *d_A, T *d_B, T *d_tRowSumMatrix) {
  if (isPow2) {
    switch (threads) {
    case 1024:
      reduceRowSumKernel<T, true, 1024>
          <<<gridSize, blockSize, sh_memSize, stream>>>(
              rowElems, paddingCols, d_A, d_B, d_tRowSumMatrix);
      break;
    case 512:
      reduceRowSumKernel<T, true, 512>
          <<<gridSize, blockSize, sh_memSize, stream>>>(
              rowElems, paddingCols, d_A, d_B, d_tRowSumMatrix);
      break;
    case 256:
      reduceRowSumKernel<T, true, 256>
          <<<gridSize, blockSize, sh_memSize, stream>>>(
              rowElems, paddingCols, d_A, d_B, d_tRowSumMatrix);
      break;
    case 128:
      reduceRowSumKernel<T, true, 128>
          <<<gridSize, blockSize, sh_memSize, stream>>>(
              rowElems, paddingCols, d_A, d_B, d_tRowSumMatrix);
      break;
    case 64:
      reduceRowSumKernel<T, true, 64>
          <<<gridSize, blockSize, sh_memSize, stream>>>(
              rowElems, paddingCols, d_A, d_B, d_tRowSumMatrix);
      break;
    case 32:
      reduceRowSumKernel<T, true, 32>
          <<<gridSize, blockSize, sh_memSize, stream>>>(
              rowElems, paddingCols, d_A, d_B, d_tRowSumMatrix);
      break;
    case 16:
      reduceRowSumKernel<T, true, 16>
          <<<gridSize, blockSize, sh_memSize, stream>>>(
              rowElems, paddingCols, d_A, d_B, d_tRowSumMatrix);
      break;
    case 8:
      reduceRowSumKernel<T, true, 8>
          <<<gridSize, blockSize, sh_memSize, stream>>>(
              rowElems, paddingCols, d_A, d_B, d_tRowSumMatrix);
      break;
    case 4:
      reduceRowSumKernel<T, true, 4>
          <<<gridSize, blockSize, sh_memSize, stream>>>(
              rowElems, paddingCols, d_A, d_B, d_tRowSumMatrix);
      break;
    case 2:
      reduceRowSumKernel<T, true, 2>
          <<<gridSize, blockSize, sh_memSize, stream>>>(
              rowElems, paddingCols, d_A, d_B, d_tRowSumMatrix);
      break;
    case 1:
      reduceRowSumKernel<T, true, 1>
          <<<gridSize, blockSize, sh_memSize, stream>>>(
              rowElems, paddingCols, d_A, d_B, d_tRowSumMatrix);
      break;
    }
  } else {
    switch (threads) {
    case 1024:
      reduceRowSumKernel<T, false, 1024>
          <<<gridSize, blockSize, sh_memSize, stream>>>(
              rowElems, paddingCols, d_A, d_B, d_tRowSumMatrix);
      break;
    case 512:
      reduceRowSumKernel<T, false, 512>
          <<<gridSize, blockSize, sh_memSize, stream>>>(
              rowElems, paddingCols, d_A, d_B, d_tRowSumMatrix);
      break;
    case 256:
      reduceRowSumKernel<T, false, 256>
          <<<gridSize, blockSize, sh_memSize, stream>>>(
              rowElems, paddingCols, d_A, d_B, d_tRowSumMatrix);
      break;
    case 128:
      reduceRowSumKernel<T, false, 128>
          <<<gridSize, blockSize, sh_memSize, stream>>>(
              rowElems, paddingCols, d_A, d_B, d_tRowSumMatrix);
      break;
    case 64:
      reduceRowSumKernel<T, false, 64>
          <<<gridSize, blockSize, sh_memSize, stream>>>(
              rowElems, paddingCols, d_A, d_B, d_tRowSumMatrix);
      break;
    case 32:
      reduceRowSumKernel<T, false, 32>
          <<<gridSize, blockSize, sh_memSize, stream>>>(
              rowElems, paddingCols, d_A, d_B, d_tRowSumMatrix);
      break;
    case 16:
      reduceRowSumKernel<T, false, 16>
          <<<gridSize, blockSize, sh_memSize, stream>>>(
              rowElems, paddingCols, d_A, d_B, d_tRowSumMatrix);
      break;
    case 8:
      reduceRowSumKernel<T, false, 8>
          <<<gridSize, blockSize, sh_memSize, stream>>>(
              rowElems, paddingCols, d_A, d_B, d_tRowSumMatrix);
      break;
    case 4:
      reduceRowSumKernel<T, false, 4>
          <<<gridSize, blockSize, sh_memSize, stream>>>(
              rowElems, paddingCols, d_A, d_B, d_tRowSumMatrix);
      break;
    case 2:
      reduceRowSumKernel<T, false, 2>
          <<<gridSize, blockSize, sh_memSize, stream>>>(
              rowElems, paddingCols, d_A, d_B, d_tRowSumMatrix);
      break;
    case 1:
      reduceRowSumKernel<T, false, 1>
          <<<gridSize, blockSize, sh_memSize, stream>>>(
              rowElems, paddingCols, d_A, d_B, d_tRowSumMatrix);
      break;
    }
  }
}

#define MAX_NUM_STREAMS 16 // 用于处理行方向的最大分块数

// convert a linear index to a row index
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
    const cudaStream_t &stream) // thrust::universal_vector
{
  if (row_sums.size() != rows) {
    row_sums.clear();
    row_sums.resize(rows);
  }

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

/*
 * 将 A 和 B 分块，每个 block 处理一部分的求和值，
 * 即 A 被分为 rowElems 个行方向的 block，B 被分为 n(x_gridSize) 个列方向的
 * block 每个行方向的 block大小为 x_blockSize，每个列方向的 block 大小为
 * y_blockSize
 * 每个行方向的 block，
 * 都计算对应的 \sum func(ai, bj)(i:1->x_blockSize，j:1->y_blockSize)
 * 最后得到 d_tRowSumMatrix(rowElems, n)
 *
 * 这个版本的 row reduce sum，是对每行的所有列进行一次性计算的
 * 所以当矩阵列数过大时，就可能导致无法进行计算（计算资源不够）
 * 所以应想办法把输入矩阵按列维度拆开成多个小矩阵，分开计算每个小矩阵的每行和再加起来合并
 */
template <typename T>
void execMyReduce(const cudaDeviceProp &prop, const cudaStream_t &stream,
                  const int &rowElems, const int &paddingCols, T *d_A, T *d_B,
                  thrust::device_vector<T> &d_value) {
  int x_blockSize = 0, y_blockSize = 16; // x操纵B，y操纵A
  int x_gridSize = 0, y_gridSize = (rowElems + y_blockSize - 1) / y_blockSize;

  getBlocksAndThreadsNum(prop, paddingCols, 128, 1024 / y_blockSize, x_gridSize,
                         x_blockSize);
  dim3 blockSize(x_blockSize, y_blockSize, 1);
  dim3 gridSize(x_gridSize, y_gridSize, 1);

  unsigned int x_paddingGridSize = PADDING(x_gridSize);
  unsigned int t_rowSumMatrixSize =
      rowElems *
      x_paddingGridSize; // 分配时需要padding后的cols，单纯是为了用于后续重复计算
                         // row reduce sum
  thrust::device_vector<T> d_tRowSumMatrix(t_rowSumMatrixSize, (T).0);
  int sh_memSize = sizeof(T) * y_blockSize *
                   ((x_blockSize / 32) + 1); // +1 for avoiding bank conflicts
  bool flag = isPow2(paddingCols);

  // d_tRowSumMatrix 为 row reduce sum 的结果，其实际不含 0 的数据维度为: elems
  // * x_gridSize，而不是 elems * x_paddingGridSize
  switchKernel<T>(flag, x_blockSize, gridSize, blockSize, sh_memSize, stream,
                  rowElems, paddingCols, d_A, d_B,
                  d_tRowSumMatrix.data().get());
  getLastCudaError("Kernel: 'reduceRowSumKernel' execution failed");

  int resCols = x_gridSize;
  // std::cout << "resCols = " << resCols << std::endl;
  if (resCols > 1) {
    thrust::device_vector<T> rowSums;
    launch_ThrustRowSumReduce<T>(rowElems, resCols, d_tRowSumMatrix, rowSums,
                                 stream);
    d_value = rowSums;
  } else {
    CUDA_CHECK(cudaMemcpyAsync(
        d_value.data().get(), d_tRowSumMatrix.data().get(),
        sizeof(T) * rowElems, cudaMemcpyDeviceToDevice, stream));
  }
}

/*
 * 两个输入 vector（向量）A(rows)和B(cols)，输出向量res(rows)
 * res(i) = \sum func(ai, bj)(j:1->cols)
 * 本质就是矩阵的行求和
 */
template <typename T>
void testMulSum(const unsigned int &rows, const unsigned int &cols,
                const std::vector<T> &A, const std::vector<T> &B,
                std::vector<T> &res) {
  cudaDeviceProp prop;
  int device = getMaxComputeDevice();
  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

  if (res.size() != rows)
    res.resize(rows);

  // thrust::device_vector<T> d_A = A;
  // thrust::device_vector<T> d_B = B;
  unsigned int paddingCols = PADDING(cols);
  T *d_B = nullptr;
  CUDA_CHECK(cudaMalloc((void **)&d_B, sizeof(T) * paddingCols));
  CUDA_CHECK(
      cudaMemcpy(d_B, B.data(), sizeof(T) * cols, cudaMemcpyHostToDevice));

  cudaStream_t streams[MAX_NUM_STREAMS];
  for (int i = 0; i < MAX_NUM_STREAMS; ++i)
    CUDA_CHECK(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));

  for (int i = 0; i < MAX_NUM_STREAMS; ++i) {
    int a_elems = (rows + MAX_NUM_STREAMS - 1) / MAX_NUM_STREAMS;
    int a_offset = i * a_elems;
    a_elems = a_offset + a_elems > rows ? rows - a_offset : a_elems;

    T *d_A = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&d_A, sizeof(T) * a_elems));
    CUDA_CHECK(cudaMemcpyAsync(d_A, A.data() + a_offset, sizeof(T) * a_elems,
                               cudaMemcpyHostToDevice, streams[i]));

    thrust::device_vector<T> d_res(a_elems);

    execMyReduce<T>(prop, streams[i], a_elems, paddingCols, d_A, d_B, d_res);

    CUDA_CHECK(cudaMemcpyAsync(res.data() + a_offset, d_res.data().get(),
                               sizeof(double) * a_elems, cudaMemcpyDeviceToHost,
                               streams[i]));
  }

  for (int i = 0; i < MAX_NUM_STREAMS; i++)
    cudaStreamSynchronize(streams[i]);
  for (int i = 0; i < MAX_NUM_STREAMS; ++i)
    CUDA_CHECK(cudaStreamDestroy(streams[i]));
}

const unsigned int rows = 4194304;
const unsigned int cols = 4194304;

int main() {
  std::vector<double> A(rows, 1), B(cols, 1);
  std::vector<double> res(rows, 0);

  using namespace std::chrono;
  // time_point<system_clock> start, end;
  // start = system_clock::now();
  // testMulSum<double>(rows, cols, A, B, res);
  // end = system_clock::now();
  // duration<double> elapsed_seconds = end - start;
  // std::time_t end_time = system_clock::to_time_t(end);
  // std::cout << "-- CUDA Elapsed time: " << elapsed_seconds.count() << " s\n"
  //           << "-- Finished computation at " << std::ctime(&end_time)
  //           << std::endl;

  time_point<system_clock> start, end;
  start = system_clock::now();
  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      res[i] += A[i] * B[j];
    }
  }
  end = system_clock::now();
  duration<double> elapsed_seconds = end - start;
  std::time_t end_time = system_clock::to_time_t(end);
  std::cout << "-- CPU Elapsed time: " << elapsed_seconds.count() << " s\n"
            << "-- Finished computation at " << std::ctime(&end_time)
            << std::endl;

  const double realVal = (double)cols;
  for (size_t i = 0; i < res.size(); ++i) {
    const auto &val = res[i];
    if (val != realVal) {
      std::cout << "Error: i = " << i << ", val = " << val << std::endl;
    }
  }

  return 0;
}