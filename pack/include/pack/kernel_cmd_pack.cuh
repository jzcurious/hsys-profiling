#pragma once

#include "pack.cuh"

#include <algorithm>
#include <cmath>
#include <core/vector_view.cuh>
#include <task/cmd.cuh>
#include <task/task.cuh>
#include <variant>

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

__global__ void kernel_cmd(PackKind auto task_pack) {
  constexpr auto overloaded = OverloadSet{};
  for (auto& t : task_pack.tasks()) std::visit(overloaded, t);
}

}  // namespace hsys::kernels::pack

namespace hsys {

template <class... TaskT>
  requires(sizeof...(TaskT) > 0)
void run_pack_pack(const TaskT&... task) {
  constexpr int block = 128;
  float max_size = std::max({std::get<0>(task.params()).size()...});
  const unsigned grid = std::ceil(max_size / block);
  kernels::pack::kernel_cmd<<<grid, block>>>(Pack(task...));
}

}  // namespace hsys
