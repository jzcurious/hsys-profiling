#pragma once

#include "utils.cuh"

#include <core/vector_view.cuh>

namespace hsys {

constexpr unsigned warp_size = 32;

__device__ float warp_reduce(float val) {
#pragma unroll
  for (uint i = warp_size / 2; i > 0; i /= 2) {
    val += __shfl_down_sync(warp_size - 1, val, i);
  }
  return val;
}

template <uint block_size>
__device__ float block_reduce(float val) {
  constexpr uint warp_num = block_size / warp_size;
  __shared__ float partial_sum[warp_num];  // NOLINT
  uint warp_id = threadIdx.x / warp_size;
  uint lane_id = threadIdx.x % warp_size;

  val = warp_reduce(val);
  if (lane_id == 0) partial_sum[warp_id] = val;
  __syncthreads();

  if (warp_id == 0)
    val = warp_reduce((lane_id < warp_num) ? partial_sum[lane_id] : float{0});

  return val;
}

template <uint block_size>
__global__ void dot_nlog(
    VectorView<float> c, const VectorView<float> a, const VectorView<float> b) {
  uint tid = internal::thread_id_1d();

  auto part_blockwide
      = block_reduce<block_size>(tid < a.size() ? a[tid] * b[tid] : float{0});
  __syncthreads();

  if (tid < a.size() and threadIdx.x == 0) atomicAdd(&c[0], part_blockwide);
}

}  // namespace hsys
