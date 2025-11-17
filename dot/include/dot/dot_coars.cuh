#pragma once

#include <core/kinds.cuh>
#include <core/vector.cuh>

namespace hsys::kernels {

template <unsigned portion>
__global__ void dot_coars(
    VectorViewK auto c, const VectorViewK auto a, const VectorViewK auto b) {
  auto begin_idx = blockDim.x * blockIdx.x + threadIdx.x * portion;
  typename decltype(c)::atom_t part{0};

#pragma unroll
  for (unsigned i = 0; i < portion; ++i) {
    unsigned j = begin_idx + i;
    if (j < a.size()) part += a[j] * b[j];
  }

  atomicAdd(&c[0], part);
}

}  // namespace hsys::kernels

namespace hsys {

template <unsigned portion, VectorK VectorT = hsys::Vector<float>>
void dot_coars(VectorT& c, const VectorT& a, const VectorT& b) {
  constexpr unsigned block = 32;
  const unsigned grid = std::ceil(static_cast<float>(a.size()) / (block * portion));
  hsys::kernels::dot_coars<portion><<<grid, block>>>(c.view(), a.view(), b.view());
}

}  // namespace hsys
