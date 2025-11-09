#pragma once

#include "context.cuh"

#include <core/kinds.cuh>
#include <core/vector.cuh>

namespace hsys::kernels {

template <AtomK AtomT>
using VAddContext = Context<VectorView<AtomT>, VectorView<AtomT>, VectorView<AtomT>>;

template <AtomK AtomT>
using VAddSlot = typename VAddContext<AtomT>::slot_t;

template <AtomK AtomT>
__global__ void vadd_persist(VAddSlot<AtomT>* slot) {
  auto tid = threadIdx.x;

  while (not slot->is_expired()) {
    if (slot->is_empty()) continue;
    __syncthreads();

    auto [c, a, b] = *slot->args();
    auto n = a.size();

    for (std::size_t offset = 0; offset < n; offset += blockDim.x) {
      auto i = tid + offset;
      if (i < n) c[i] = a[i] + b[i];
    }

    __syncthreads();
    if (tid == 0) slot->clear();
  }
}

}  // namespace hsys::kernels

namespace hsys {

template <AtomK AtomT>
void vadd_persist(Vector<AtomT>& c, Vector<AtomT>& a, Vector<AtomT>& b) {
  constexpr unsigned block = 1024;
  static kernels::VAddContext<AtomT> ctx;
  static bool first_call = true;

  ctx.set_args(c.view(), a.view(), b.view());

  if (first_call) {
    hsys::kernels::vadd_persist<AtomT><<<1, block, 0, ctx.stream()>>>(ctx.slot());
    first_call = false;
  }

  while (not ctx.slot()->is_empty());
}

}  // namespace hsys
