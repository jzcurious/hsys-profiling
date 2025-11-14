#pragma once

#include <algorithm>
#include <cmath>
#include <core/vector_view.cuh>
#include <task/cmd.cuh>

namespace hsys::kernels::var {

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

template <class... TaskT>
__global__ void kernel_cmd(TaskT... task_pack) {
  auto tid = blockIdx.x * blockDim.x + threadIdx.x;

  auto do_task_grid = [tid](auto& task) {
    auto [c, a, b] = task.params();
    auto n = a.size();
    using cmd_t = typename std::remove_reference_t<decltype(task)>::command_t;
    if (tid < n) c[tid] = do_task_thread(a[tid], b[tid], cmd_t{});
    __syncthreads();
  };

  ((do_task_grid(task_pack)), ...);
}

template <class... TaskT>
__global__ void kernel_cmd_noop(TaskT... task_pack) {}

}  // namespace hsys::kernels::var

namespace hsys {

template <class... TaskT>
  requires(sizeof...(TaskT) > 0)
void run_pack_var(const TaskT&... task) {
  constexpr int block = 128;
  float max_size = std::max({std::get<0>(task.params()).size()...});
  const unsigned grid = std::ceil(max_size / block);
  kernels::var::kernel_cmd<<<grid, block>>>(task...);
}

template <class... TaskT>
  requires(sizeof...(TaskT) > 0)
void run_pack_var_noop(const TaskT&... task) {
  constexpr int block = 128;
  float max_size = std::max({std::get<0>(task.params()).size()...});
  const unsigned grid = std::ceil(max_size / block);
  kernels::var::kernel_cmd_noop<<<grid, block>>>(task...);
}

}  // namespace hsys
