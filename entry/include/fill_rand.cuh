#include <chrono>
#include <core/kinds.cuh>
#include <curand.h>

template <class T>
  requires(hsys::VectorK<T> or hsys::MatrixK<T>)
inline void fill_rand(T& tensor, float mean = 0.0f, float stddev = 1.0f) {
  curandGenerator_t gen;  // NOLINT
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);

  auto now = std::chrono::steady_clock::now();
  unsigned long long seed = now.time_since_epoch().count();

  curandSetPseudoRandomGeneratorSeed(gen, seed);
  curandGenerateNormal(gen, tensor.data().data(), tensor.size(), mean, stddev);
}
