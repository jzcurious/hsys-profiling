#pragma once

#include <tuple>

namespace hsys {

template <class SlotT>
struct Context {
  Context() {
    cudaStreamCreate(&stream_);
    cudaMalloc(&slot_, sizeof(SlotT));
    std::apply(
        [](auto*&... arg_ptr) {
          (cudaMalloc(&arg_ptr, sizeof(std::decay_t<decltype(arg_ptr)>)), ...);
        },
        slot_->args());
  }

  Context(const Context&) = delete;
  Context(Context&&) = delete;

  Context& operator=(const Context&) = delete;
  Context& operator=(Context&&) = delete;

  [[nodiscard]] cudaStream_t stream() const {
    return stream_;
  }

  [[nodiscard]] SlotT* slot() {
    return slot_;
  }

  template <class... T>
  __host__ void set_args(const T&... new_arg) {
    std::apply(
        [&](auto&... arg_ptr) {
          ((cudaMemcpy(
               arg_ptr, &new_arg, sizeof(decltype(new_arg)), cudaMemcpyHostToDevice)),
              ...);
        },
        slot_->args());
  }

  ~Context() {
    cudaStreamDestroy(stream_);
    if (slot_) {
      std::apply([](auto&... arg_ptr) { (cudaFree(arg_ptr), ...); }, slot_->args());
      cudaFree(slot_);
    }
  }

 private:
  cudaStream_t stream_ = nullptr;
  SlotT* slot_ = nullptr;
};

}  // namespace hsys
