#pragma once

#include "dot_nlog.cuh"
#include "utils.cuh"

#include <core/vector_view.cuh>

namespace hsys {

template <uint block_size>
__global__ void dot_nlog_vect(
    VectorView<float> c, const VectorView<float> a, const VectorView<float> b) {
  uint tid = internal::thread_id_1d();
  float part_thread{0};
  uint j = tid * 4;

  if (j + 3 < a.size()) {
    float4 vec_a = *reinterpret_cast<const float4*>(&a[j]);  // NOLINT
    float4 vec_b = *reinterpret_cast<const float4*>(&b[j]);  // NOLINT
    part_thread
        += vec_a.x * vec_b.x + vec_a.y * vec_b.y + vec_a.z * vec_b.z + vec_a.w * vec_b.w;
  } else {
    for (unsigned k = j; k < a.size(); ++k) part_thread += a[k] * b[k];
  }

  auto part_blockwide = block_reduce<block_size>(tid < a.size() ? part_thread : float{0});
  __syncthreads();

  if (tid < a.size() and threadIdx.x == 0) atomicAdd(&c[0], part_blockwide);
}

}  // namespace hsys
