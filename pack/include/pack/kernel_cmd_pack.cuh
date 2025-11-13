#pragma once

#include "pack.cuh"

#include <algorithm>
#include <cmath>
#include <core/vector_view.cuh>
#include <task/cmd.cuh>
#include <task/task.cuh>
#include <tuple>
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
  using cmd_t = std::remove_reference_t<decltype(task)>::command_t;
  if (tid < n) c[tid] = do_task_thread(a[tid], b[tid], cmd_t{});
  __syncthreads();
}

__global__ void kernel_cmd(PackKind auto task_pack) {
  struct {
    __device__ void operator()(
        Task<Add, VectorView<float>, VectorView<float>, VectorView<float>>& t) {
      do_task_grid(t);
    }

    __device__ void operator()(
        Task<Sub, VectorView<float>, VectorView<float>, VectorView<float>>& t) {
      do_task_grid(t);
    }

    __device__ void operator()(
        Task<Mul, VectorView<float>, VectorView<float>, VectorView<float>>& t) {
      do_task_grid(t);
    }

    __device__ void operator()(
        Task<Div, VectorView<float>, VectorView<float>, VectorView<float>>& t) {
      do_task_grid(t);
    }
  } overload_set;

  for (auto t : task_pack.tasks()) std::visit(overload_set, t);
}

}  // namespace hsys::kernels::pack

namespace hsys {

template <class... TaskT>
void run_pack_pack(const TaskT&... task) {
  auto pack = Pack(task...);
  constexpr int block = 128;
  auto max_size = std::max({std::get<0>(task.params()).size()...});
  unsigned const grid = std::ceil(static_cast<float>(max_size) / block);
  kernels::pack::kernel_cmd<<<grid, block>>>(pack);
}

}  // namespace hsys
