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

  ~Slot() = default;

 private:
  bool expired_ = false;
  std::tuple<ArgT*...> args_{};
};

}  // namespace hsys
