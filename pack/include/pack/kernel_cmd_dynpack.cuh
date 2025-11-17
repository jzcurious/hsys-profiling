#pragma once

#include "dynpack.cuh"

#include <cmath>
#include <core/vector_view.cuh>
#include <cuda/std/variant>
#include <task/cmd.cuh>
#include <task/task.cuh>

namespace hsys::kernels::pack {

template <class AtomT>
__device__ AtomT do_task_thread(AtomT a, AtomT b, Add) {
  return a + b;
}

template <class AtomT>
__device__ AtomT do_task_thread(AtomT a, AtomT b, Sub) {
  return a - b;
}

template <class AtomT>
__device__ AtomT do_task_thread(AtomT a, AtomT b, Mul) {
  return a * b;
}

template <class AtomT>
__device__ AtomT do_task_thread(AtomT a, AtomT b, Div) {
  return a / b;
}

__device__ void do_task_grid(auto& task) {
  auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  auto [c, a, b] = task.params();
  auto n = a.size();
  using cmd_t = typename std::remove_reference_t<decltype(task)>::command_t;
  if (tid < n) c[tid] = do_task_thread(a[tid], b[tid], cmd_t{});
  __syncthreads();
}

struct OverloadSet {
  template <class T>
  __device__ void operator()(
      Task<T, VectorView<float>, VectorView<float>, VectorView<float>>& t) const {
    do_task_grid(t);
  }
};

__global__ void kernel_cmd(DynPackKind auto task_pack) {
  constexpr auto overloaded = OverloadSet{};
  for (unsigned i = 0; i < task_pack.size(); ++i)
    cuda::std::visit(overloaded, task_pack[i]);
}

}  // namespace hsys::kernels::pack

namespace hsys {

void run_pack_dynpack(const DynPackKind auto& task_pack) {
  constexpr int block = 128;
  float max_size = 0;
  for (unsigned i = 0; i < task_pack.size(); ++i) {
    unsigned s = 0;
    cuda::std::visit(
        [&s](auto&& t) { s = std::get<0>(t.params()).size(); }, task_pack[i]);
    if (max_size < s) max_size = s;  // NOLINT
  }
  const unsigned grid = std::ceil(max_size / block);
  kernels::pack::kernel_cmd<<<grid, block>>>(task_pack);
}

}  // namespace hsys
