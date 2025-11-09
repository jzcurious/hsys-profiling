#pragma once

#include <core/kinds.cuh>
#include <tuple>

namespace hsys {

template <ViewK... ArgT>
struct Slot {
  Slot(const Slot&) = delete;
  Slot(Slot&&) = delete;

  Slot& operator=(const Slot&) = delete;
  Slot& operator=(Slot&&) = delete;

  [[nodiscard]] __host__ __device__ bool is_empty() const {
    return args_;
  }

  __host__ __device__ void set_args(std::tuple<ArgT...>* new_args) {
    args_ = new_args;
  }

  __host__ __device__ std::tuple<ArgT...>* args() {
    return args_;
  }

  __host__ __device__ void clear() {
    args_ = nullptr;
  }

  __host__ __device__ void expire() {
    expired_ = true;
  }

  __host__ __device__ bool is_expired() const {
    return expired_;
  }

  ~Slot() = default;

 private:
  bool expired_ = false;
  std::tuple<ArgT...>* args_ = nullptr;
};

}  // namespace hsys
