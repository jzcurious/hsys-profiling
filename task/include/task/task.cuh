#pragma once

#include <tuple>

namespace hsys {

template <class CommandT, class... ParameterT>
struct Task {
  using command_t = CommandT;
  using params_t = std::tuple<ParameterT...>;

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
