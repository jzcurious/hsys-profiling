#pragma once

#include <array>
#include <variant>

namespace hsys {

template <unsigned size, class... TaskT>
  requires(sizeof...(TaskT) > 0)
struct Pack {
  struct hsys_pack_kind {};

  using common_task_t = std::variant<TaskT...>;
  using tasks_t = std::array<common_task_t, size>;

  template <class... T>
  __host__ Pack(const T&... task)
      : tasks_{task...} {}

  __host__ __device__ tasks_t& tasks() {
    return tasks_;
  }

 private:
  tasks_t tasks_;
};

template <class... T>
__host__ Pack(const T&... task) -> Pack<sizeof...(T), T...>;

template <class T>
concept PackKind = requires { typename T::hsys_pack_kind; };

}  // namespace hsys
