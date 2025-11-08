#pragma once

#include <core/kinds.cuh>
#include <core/vector.cuh>

namespace hsys::kernels {

__global__ void vadd(
    VectorViewK auto c, const VectorViewK auto a, const VectorViewK auto b) {
  auto tid = blockDim.x * blockIdx.x + threadIdx.x;
  c[tid] += a[tid] + b[tid];
}

}  // namespace hsys::kernels

namespace hsys {

template <VectorK VectorT = hsys::Vector<float>>
void vadd(VectorT& c, const VectorT& a, const VectorT& b) {
  constexpr unsigned block = 128;
  const unsigned grid = std::ceil(a.size() / block);
  hsys::kernels::vadd<<<grid, block>>>(c.view(), a.view(), b.view());
}

}  // namespace hsys
