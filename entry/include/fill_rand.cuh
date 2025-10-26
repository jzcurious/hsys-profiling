#include <chrono>
#include <core/vector.cuh>
#include <curand.h>

inline void fill_rand(
    hsys::Vector<float>& vector, float mean = 0.0f, float stddev = 1.0f) {
  curandGenerator_t gen;  // NOLINT
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);

  auto now = std::chrono::steady_clock::now();
  unsigned long long seed = now.time_since_epoch().count();

  curandSetPseudoRandomGeneratorSeed(gen, seed);
  curandGenerateNormal(gen, vector.data().data(), vector.size(), mean, stddev);
}
