#include <thrust/device_vector.h>

struct transformA {
  double lambda;
  double val;
  __host__ __device__ transformA(const double &_lambda, const double &_val)
      : lambda(_lambda), val(_val) {}
  __host__ __device__ double operator()(const double &x) const {
    return lambda * x + val;
  }
};

int main() { return 0; }