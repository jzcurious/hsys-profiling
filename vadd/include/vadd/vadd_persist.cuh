#pragma once

#include "slot.cuh"
#include <core/kinds.cuh>
#include <core/vector.cuh>

namespace hsys::kernels {

template <AtomK AtomT>
using VAddSlot
    = Slot<VectorView<AtomT>, const VectorView<AtomT>, const VectorView<AtomT>>;

template <AtomK AtomT>
__global__ void vadd_persist(VAddSlot<AtomT>* slot) {
  auto tid = threadIdx.x;

  while (not slot->is_expired()) {
    if (slot->is_empty()) continue;
    __syncthreads();

    auto [c, a, b] = slot->args();
    auto size = a->size();

    for (std::size_t offset = 0; offset < size; offset += blockDim.x) {
      auto i = tid + offset;
      if (i < size) (*c)[i] = (*a)[i] + (*b)[i];
    }

    __syncthreads();
    if (tid == 0) slot->clear();
  }
}

}  // namespace hsys::kernels

namespace hsys {

template <AtomK AtomT>
void vadd_persist(Vector<AtomT>& c, const Vector<AtomT>& a, const Vector<AtomT>& b) {
  constexpr unsigned block = 1024;
  static kernels::VAddSlot<AtomT> slot;
  static bool first_call = true;

  slot(&c.view(), &a.view(), &b.view());

  if (first_call) {
    hsys::kernels::vadd_persist<<<1, block, 0, slot.stream()>>>(&slot);
    first_call = false;
  }

  while (not slot.is_empty());
}

}  // namespace hsys
