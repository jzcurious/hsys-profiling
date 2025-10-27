#pragma once

#include "utils.cuh"

#include <core/vector_view.cuh>

namespace hsys {

template <unsigned portion>
  requires(portion % 4 == 0)
__global__ void dot_vectorized(
    VectorView<float> c, const VectorView<float> a, const VectorView<float> b) {
  auto begin_idx = internal::thread_id_1d() * portion;
  float part{0};

  constexpr unsigned nvec = portion / 4;

#pragma unroll
  for (unsigned i = 0; i < nvec; ++i) {
    const unsigned j = begin_idx + i * 4;
    if (j + 3 < a.size()) {
      float4 vec_a = *reinterpret_cast<const float4*>(&a[j]);  // NOLINT
      float4 vec_b = *reinterpret_cast<const float4*>(&b[j]);  // NOLINT
      part += vec_a.x * vec_b.x + vec_a.y * vec_b.y + vec_a.z * vec_b.z
              + vec_a.w * vec_b.w;
    } else {
      for (unsigned k = j; k < a.size(); ++k) part += a[k] * b[k];
    }
  }

  atomicAdd(&c[0], part);
}

}  // namespace hsys
