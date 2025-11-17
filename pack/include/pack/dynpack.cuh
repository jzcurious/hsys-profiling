#pragma once

#include "pack.cuh"
#include "uvariant.cuh"

#include <task/task.cuh>

namespace hsys {

template <class T>
concept DynPackKind = requires { typename T::hsys_dynpack_kind; };

template <unsigned capacity, class TaskVarT>
struct DynPack {
  struct hsys_dynpack_kind {};

  using common_task_t = TaskVarT;
  using pack_t = Pack<capacity, TaskVarT>;

  template <TaskKind... T>
  __host__ DynPack(const T&... task) {
    (push(task), ...);
  }

  [[nodiscard]] __host__ __device__ constexpr std::size_t bulk() const {
    return capacity;
  }

  [[nodiscard]] __host__ __device__ std::size_t size() const {
    return size_;
  }

  [[nodiscard]] __host__ __device__ bool is_full() const {
    return capacity == size_;
  }

  __host__ void push(const TaskVarT& task) {
    if (is_full()) return;
    pack_[size_++] = task;
  }

  __host__ __device__ TaskVarT& operator[](std::size_t i) {
    return pack_[i];
  }

  __host__ __device__ const TaskVarT& operator[](std::size_t i) const {
    return pack_[i];
  }

 private:
  std::size_t size_ = 0;
  Pack<capacity, TaskVarT> pack_;
};

template <TaskKind... T>
__host__ __device__ DynPack(const T&... task)
    -> DynPack<2 * sizeof...(T), make_unique_variant_t<T...>>;

}  // namespace hsys
