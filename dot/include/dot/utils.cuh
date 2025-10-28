#pragma once

namespace hsys::internal {

inline __device__ unsigned thread_id_1d() {
  return blockDim.x * blockIdx.x + threadIdx.x;
}

}  // namespace hsys::internal
