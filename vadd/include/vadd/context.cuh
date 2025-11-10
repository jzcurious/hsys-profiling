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
    cudaMallocManaged(&slot_, sizeof(Slot<ArgT...>));
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

  template <ViewK... T>
  __host__ void set_args(const T&... new_arg) {
    auto new_args = std::make_tuple(new_arg...);
    cudaMemcpyAsync(args_, &new_args, sizeof(new_args), cudaMemcpyHostToDevice, stream_);
    slot_->set_args(args_);
  }

  ~Context() {
    if (slot_) {
      slot_->expire();
      cudaFree(slot_);
    }
    if (args_) cudaFree(args_);
    if (stream_) cudaStreamDestroy(stream_);
  }

 private:
  cudaStream_t stream_ = nullptr;
  Slot<ArgT...>* slot_ = nullptr;
  std::tuple<ArgT...>* args_ = nullptr;
};

}  // namespace hsys
