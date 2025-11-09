#pragma once

#include <core/data.cuh>
#include <core/kinds.cuh>
#include <tuple>

#include "slot.cuh"

namespace hsys {

template <ViewK... ArgT>
struct Context {
  using slot_t = Slot<ArgT...>;

  Context() {
    cudaStreamCreate(&stream_);
    cudaMalloc(&slot_, sizeof(Slot<ArgT...>));
    cudaMalloc(&args_, sizeof(std::tuple<ArgT...>));
  }

  Context(const Context&) = delete;
  Context(Context&&) = delete;

  Context& operator=(const Context&) = delete;
  Context& operator=(Context&&) = delete;

  [[nodiscard]] cudaStream_t stream() const {
    return stream_;
  }

  [[nodiscard]] Slot<ArgT...>* slot() {
    return slot_;
  }

  template <class... T>
  __host__ void set_args(const T&... new_arg) {
    auto new_args = std::make_tuple(new_arg...);
    cudaMemcpy(args_, &new_args, sizeof(new_args), cudaMemcpyHostToDevice);
    [&]<std::size_t... i>(std::index_sequence<i...>) {
      ((std::get<i>(slot_->args()) = &std::get<i>(*args_)), ...);
    }(std::make_index_sequence<sizeof...(new_arg)>{});
  }

  ~Context() {
    if (stream_) cudaStreamDestroy(stream_);
    if (args_) cudaFree(args_);
    if (slot_) cudaFree(slot_);
  }

 private:
  cudaStream_t stream_ = nullptr;
  Slot<ArgT...>* slot_ = nullptr;
  std::tuple<ArgT...>* args_ = nullptr;
};

}  // namespace hsys
