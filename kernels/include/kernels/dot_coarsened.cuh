#pragma once

#include "utils.cuh"

#include <core/kinds.cuh>

namespace hsys {

template <unsigned portion>
__global__ void dot_coarsened(
    VectorViewK auto c, VectorViewK auto a, VectorViewK auto b) {
  auto begin_idx = internal::thread_id_1d() * portion;
  typename decltype(c)::atom_t part{0};

#pragma unroll
  for (unsigned i = 0; i < portion; ++i) {
    unsigned j = begin_idx + i;
    if (j < a.size()) part += a[j] * b[j];
  }

  atomicAdd(&c[0], part);
}

}  // namespace hsys
