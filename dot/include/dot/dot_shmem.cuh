#pragma once

#include <core/vector.cuh>
#include <core/vector_view.cuh>

namespace hsys::kernels {

template <unsigned portion>
  requires(portion % 4 == 0)
__global__ void dot_shmem(
    VectorView<float> c, const VectorView<float> a, const VectorView<float> b) {

  auto begin_idx = blockDim.x * blockIdx.x + threadIdx.x * portion;
  float part_thread{0};
  constexpr unsigned nvec = portion / 4;

#pragma unroll
  for (unsigned i = 0; i < nvec; ++i) {
    const unsigned j = begin_idx + i * 4;
    if (j + 3 < a.size()) {
      float4 vec_a = *reinterpret_cast<const float4*>(&a[j]);  // NOLINT
      float4 vec_b = *reinterpret_cast<const float4*>(&b[j]);  // NOLINT
      part_thread += vec_a.x * vec_b.x + vec_a.y * vec_b.y + vec_a.z * vec_b.z
                     + vec_a.w * vec_b.w;
    } else {
      for (unsigned k = j; k < a.size(); ++k) part_thread += a[k] * b[k];
    }
  }

  __shared__ float part_blockwide[1];  // NOLINT
  if (threadIdx.x == 0) part_blockwide[0] = 0.0f;
  __syncthreads();

  atomicAdd(part_blockwide, part_thread);  // NOLINT
  __syncthreads();

  if (threadIdx.x == 0) atomicAdd(&c[0], part_blockwide[0]);
}

}  // namespace hsys::kernels

namespace hsys {

template <unsigned portion, VectorK VectorT = hsys::Vector<float>>
void dot_shmem(VectorT& c, const VectorT& a, const VectorT& b) {
  constexpr unsigned block = 32;
  const unsigned grid = std::ceil(static_cast<float>(a.size()) / (block * portion));
  hsys::kernels::dot_shmem<portion><<<grid, block>>>(c.view(), a.view(), b.view());
}

}  // namespace hsys
