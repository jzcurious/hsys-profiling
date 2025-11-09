#pragma once

#include <core/kinds.cuh>
#include <core/vector.cuh>

namespace hsys {

template <AtomK AtomT>
struct VAddTask {

  VAddTask() {
    cudaStreamCreate(&stream_);
    cudaMallocManaged(&processed_ptr_, sizeof(bool));
    cudaMallocManaged(&destroyed_ptr_, sizeof(bool));
    *processed_ptr_ = true;
    *destroyed_ptr_ = false;
  }

  VAddTask(const VAddTask&) = delete;
  VAddTask& operator=(const VAddTask&) = delete;

  VAddTask(VAddTask&&) = delete;
  VAddTask& operator=(VAddTask&&) = delete;

  __host__ __device__ std::size_t size() const {
    return a_->size();
  }

  __host__ cudaStream_t stream() const {
    return stream_;
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

  [[nodiscard]] __host__ __device__ bool is_destroyed() const {
    return destroyed_ptr_ and *destroyed_ptr_;
  }

  [[nodiscard]] __host__ __device__ bool is_processed() const {
    return processed_ptr_ and *processed_ptr_;
  }

  [[nodiscard]] __host__ __device__ bool has_args() const {
    return a_ and b_ and c_;
  }

  __host__ void set_args(
      VectorView<AtomT>* c, const VectorView<AtomT>* a, const VectorView<AtomT>* b) {
    if (is_destroyed()) return;
    a_ = a;
    b_ = b;
    c_ = c;
  }

  [[nodiscard]] __host__ __device__ bool is_issued() const {
    return (not is_destroyed()) and (not is_processed()) and has_args();
  }

  __host__ bool issue() {
    bool ready_for_issue = (not is_destroyed()) and has_args();
    if (ready_for_issue) *processed_ptr_ = false;
    return ready_for_issue;
  }

  __device__ bool complete() {
    if (is_issued()) return *processed_ptr_ = true;
    return false;
  }

  __host__ void destroy() {
    *destroyed_ptr_ = true;
    cudaFree(destroyed_ptr_);
    cudaFree(processed_ptr_);
    cudaStreamDestroy(stream_);
  }

  __host__ ~VAddTask() {
    destroy();
  }

 private:
  cudaStream_t stream_ = nullptr;
  bool* destroyed_ptr_ = nullptr;
  bool* processed_ptr_ = nullptr;
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
    hsys::kernels::vadd_persist<<<1, block, 0, task.stream()>>>(&task);
    first_call = false;
  }
}

}  // namespace hsys
