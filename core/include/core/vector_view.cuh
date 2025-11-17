#pragma once

#include "kinds.cuh"

namespace hsys {

template <AtomK AtomT>
struct VectorView {
  struct hsys_vector_view_feature {};

 public:
  using atom_t = AtomT;

 private:
  atom_t* data_;
  std::size_t size_;

 public:
  __host__ __device__ VectorView()
      : data_(nullptr)
      , size_(0) {}

  __host__ __device__ VectorView(atom_t* data, std::size_t size)
      : data_(data)
      , size_(size) {}

  VectorView(const VectorView& other) = default;
  VectorView(VectorView&& other) = default;

  VectorView& operator=(const VectorView& other) = default;
  VectorView& operator=(VectorView&& other) = default;

  ~VectorView() = default;

  [[nodiscard]] __host__ __device__ std::size_t size() const {
    return size_;
  }

  __host__ __device__ atom_t& operator[](std::size_t i) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    return data_[i];
  }

  __host__ __device__ const atom_t& operator[](std::size_t i) const {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    return data_[i];
  }
};

}  // namespace hsys
