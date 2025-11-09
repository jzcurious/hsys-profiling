#pragma once

#include <core/kinds.cuh>
#include <tuple>

namespace hsys {

template <ViewK... ArgT>
struct Slot {
  Slot() {
    cudaStreamCreate(&stream_);
  }

  template <class... T>
  Slot(T&&... args)
      : args_{std::forward<T>(args)...} {
    cudaStreamCreate(&stream_);
  }

  Slot(const Slot&) = delete;
  Slot(Slot&&) = delete;

  Slot& operator=(const Slot&) = delete;
  Slot& operator=(Slot&&) = delete;

  [[nodiscard]] __host__ __device__ bool is_empty() const {
    return std::apply([](const auto&... args) { return (not(args or ...)); }, args_);
  }

  __host__ __device__ auto& args() {
    return args_;
  }

  __host__ __device__ void clear() {
    std::apply([](auto&... args) { ((args = nullptr), ...); }, args_);
  }

  __host__ __device__ void expire() {
    expired_ = true;
  }

  __host__ __device__ bool is_expired() const {
    return expired_;
  }

  __host__ cudaStream_t stream() const {
    return stream_;
  }

  template <class... T>
  __host__ void operator()(T&&... new_args) {
    std::apply(
        [&new_args...](auto&... args) { ((args = std::forward<T>(new_args)), ...); },
        args_);
  }

  ~Slot() {
    cudaStreamDestroy(stream_);
  }

 private:
  bool expired_ = false;
  cudaStream_t stream_ = nullptr;
  std::tuple<ArgT*...> args_{};
};

}  // namespace hsys
