#pragma once

#include <core/kinds.cuh>
#include <core/vector.cuh>

namespace hsys::kernels {

__global__ void dot_atomic(
    VectorViewK auto c, const VectorViewK auto a, const VectorViewK auto b) {
  auto tid = blockDim.x * blockIdx.x + threadIdx.x;
  atomicAdd(&c[0], a[tid] * b[tid]);
}

}  // namespace hsys::kernels

namespace hsys {

template <VectorK VectorT = hsys::Vector<float>>
void dot_atomic(VectorT& c, const VectorT& a, const VectorT& b) {
  constexpr unsigned block = 128;
  const unsigned grid = std::ceil(static_cast<float>(a.size()) / block);
  hsys::kernels::dot_atomic<<<grid, block>>>(c.view(), a.view(), b.view());
}

}  // namespace hsys
