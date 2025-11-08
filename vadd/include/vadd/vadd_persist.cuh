#pragma once

#include <core/kinds.cuh>
#include <core/vector.cuh>

namespace hsys {

template <AtomK AtomT>
struct VAddTask {

  VAddTask() {
    cudaStreamCreate(&stream_);
  }

  VAddTask(const VAddTask&) = delete;
  VAddTask& operator=(const VAddTask&) = delete;

  VAddTask(VAddTask&&) = delete;
  VAddTask& operator=(VAddTask&&) = delete;

  __host__ __device__ std::size_t size() const {
    return a_->size();
  }

  __host__ __device__ const AtomT& a(std::size_t i) {
    return (*a_)[i];
  }

  __host__ __device__ const AtomT& b(std::size_t i) {
    return (*b_)[i];
  }

  __host__ __device__ AtomT& c(std::size_t i) {
    return (*c_)[i];
  }

  __host__ void set_args(
      VectorView<AtomT>* c, const VectorView<AtomT>* a, const VectorView<AtomT>* b) {
    if (destroyed_) return;
    a_ = a;
    b_ = b;
    c_ = c;
  }

  [[nodiscard]] __host__ __device__ bool is_destroyed() const {
    return destroyed_;
  }

  [[nodiscard]] __host__ __device__ bool is_issued() const {
    return (not destroyed_) and (not processed_) and (a_ and b_ and c_);
  }

  __host__ bool issue() {
    bool ready_for_issue = (not destroyed_) and (a_ and b_ and c_);
    if (ready_for_issue) processed_ = false;
    return ready_for_issue;
  }

  __device__ bool complete() {
    if (is_issued()) return processed_ = true;
    return false;
  }

  __host__ void destroy() {
    destroyed_ = true;
    cudaStreamDestroy(stream_);
  }

  __host__ ~VAddTask() {
    destroy();
  }

 private:
  cudaStream_t stream_ = nullptr;
  bool processed_ = true;
  bool destroyed_ = false;
  const VectorView<AtomT>* a_ = nullptr;
  const VectorView<AtomT>* b_ = nullptr;
  VectorView<AtomT>* c_ = nullptr;
};

}  // namespace hsys

namespace hsys::kernels {

template <AtomK AtomT>
__global__ void vadd_persist(VAddTask<AtomT>* task) {
  auto tid = threadIdx.x;

  while (not task->is_destroyed()) {
    if (not task->is_issued()) continue;

    for (std::size_t offset = 0; offset < task->size(); offset += blockDim.x) {
      auto i = tid + offset;
      if (i < task->size()) task->c(i) = task->a(i) + task->b(i);
    }

    __syncthreads();
    if (tid == 0) task->complete();
  }
}

}  // namespace hsys::kernels

namespace hsys {

template <AtomK AtomT>
void vadd_persist(Vector<AtomT>& c, const Vector<AtomT>& a, const Vector<AtomT>& b) {
  constexpr unsigned block = 1024;
  static VAddTask<AtomT> task;
  static bool first_call = true;

  while (task.is_issued());
  task.set_args(&c.view(), &a.view(), &b.view());
  task.issue();

  if (first_call) {
    hsys::kernels::vadd_persist<<<1, block>>>(&task);
    first_call = false;
  }
}

}  // namespace hsys
