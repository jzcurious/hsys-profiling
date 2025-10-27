#pragma once

#include "utils.cuh"
#include "warp_size.cuh"

#include <core/vector_view.cuh>

namespace hsys {

__device__ float warp_reduce_fadd(float val) {
#pragma unroll
  for (uint i = warp_size / 2; i > 0; i /= 2) {
    val = __fadd_rn(__shfl_down_sync(warp_size - 1, val, i), val);  // NOLINT
  }
  return val;
}

template <uint block_size>
__device__ float block_reduce_fadd(float val) {
  constexpr uint warp_num = block_size / warp_size;
  __shared__ float partial_sum[warp_num];  // NOLINT
  uint warp_id = threadIdx.x / warp_size;
  uint lane_id = threadIdx.x % warp_size;

  val = warp_reduce_fadd(val);
  if (lane_id == 0) partial_sum[warp_id] = val;
  __syncthreads();

  if (warp_id == 0)
    val = warp_reduce((lane_id < warp_num) ? partial_sum[lane_id] : float{0});

  return val;
}

template <uint block_size>
__global__ void dot_nlog_vect_fma(
    VectorView<float> c, const VectorView<float> a, const VectorView<float> b) {
  uint tid = internal::thread_id_1d();
  float part_thread{0};
  uint j = tid * 4;

  if (j + 3 < a.size()) {
    float4 vec_a = *reinterpret_cast<const float4*>(&a[j]);  // NOLINT
    float4 vec_b = *reinterpret_cast<const float4*>(&b[j]);  // NOLINT
    part_thread = __fmaf_rn(vec_a.x, vec_b.x, part_thread);
    part_thread = __fmaf_rn(vec_a.y, vec_b.y, part_thread);
    part_thread = __fmaf_rn(vec_a.z, vec_b.z, part_thread);
    part_thread = __fmaf_rn(vec_a.w, vec_b.w, part_thread);
  } else {
    for (unsigned k = j; k < a.size(); ++k)
      part_thread = __fmaf_rn(a[k], b[k], part_thread);
  }

  auto part_blockwide
      = block_reduce_fadd<block_size>(tid < a.size() ? part_thread : float{0});
  __syncthreads();

  if (tid < a.size() and threadIdx.x == 0) atomicAdd(&c[0], part_blockwide);
}

}  // namespace hsys
