#pragma once

#include "task.cuh"
#include "uvariant.cuh"

#include <array>

namespace hsys {

template <class T>
concept PackKind = requires { typename T::hsys_pack_kind; };

template <unsigned size, class TaskVarT>
struct Pack {
  struct hsys_pack_kind {};

  using common_task_t = TaskVarT;
  using tasks_t = std::array<common_task_t, size>;

  Pack() = default;

  Pack(const Pack& other) = default;
  Pack(Pack&& other) = default;

  Pack& operator=(const Pack& other) = default;
  Pack& operator=(Pack&& other) = default;

  template <TaskKind... T>
  __host__ __device__ Pack(const T&... task)
      : tasks_{task...} {}

  __host__ __device__ tasks_t& tasks() {
    return tasks_;
  }

  __host__ __device__ TaskVarT& operator[](std::size_t i) {
    return tasks_[i];
  }

  __host__ __device__ const TaskVarT& operator[](std::size_t i) const {
    return tasks_[i];
  }

  ~Pack() = default;

 private:
  tasks_t tasks_;
};

template <class... T>
__host__ __device__ Pack(const T&... task)
    -> Pack<sizeof...(T), make_unique_variant_t<T...>>;

}  // namespace hsys
