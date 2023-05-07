#include "cuBLASCheck.cuh"
#include <cublas_v2.h>

int main() {
  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));
  return 0;
}