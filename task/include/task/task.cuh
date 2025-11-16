#pragma once

#include <tuple>

namespace hsys {

template <class T>
concept TaskKind = requires { typename T::hsys_task_kind; };

template <class CommandT, class... ParameterT>
struct Task {
  struct hsys_task_kind {};

  using command_t = CommandT;
  using params_t = std::tuple<ParameterT...>;

  __host__ __device__ Task() {}  // NOLINT

  template <class T, class... U>
  Task(T, const U&... params)
      : params_{params...} {}

  __host__ __device__ std::tuple<ParameterT...>& params() {
    return params_;
  }

  __host__ __device__ const std::tuple<ParameterT...>& params() const {
    return params_;
  }

 private:
  std::tuple<ParameterT...> params_;
};

template <class T, class... U>
Task(T, const U&... params) -> Task<T, U...>;

}  // namespace hsys
