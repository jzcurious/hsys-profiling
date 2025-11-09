#pragma once

namespace hsys {

template <class SlotT>
struct Context {
  Context() {
    cudaStreamCreate(&stream_);
    cudaMallocManaged(&slot_, sizeof(SlotT));
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

  ~Context() {
    cudaStreamDestroy(stream_);
    if (slot_) cudaFree(slot_);
  }

 private:
  cudaStream_t stream_ = nullptr;
  SlotT* slot_ = nullptr;
};

}  // namespace hsys
