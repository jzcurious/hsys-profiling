#pragma once

#include "utils.cuh"

#include <core/kinds.cuh>

namespace hsys {

__global__ void dot_atomic(VectorViewK auto c, VectorViewK auto a, VectorViewK auto b) {
  auto tid = internal::thread_id_1d();
  atomicAdd(&c[0], a[tid] * b[tid]);
}

}  // namespace hsys
