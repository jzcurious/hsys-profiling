#pragma once

#include <work1/kinds.cuh>

namespace hsys {

template <AtomK AtomT>
struct MatrixView {
  struct hsys_matrix_view_feature {};

 public:
  using atom_t = AtomT;

 private:
  AtomT* data_;
  std::size_t nrows_;
  std::size_t ncols_;

 public:
  __host__ __device__ MatrixView(AtomT* data, std::size_t nrows, std::size_t ncols)
      : data_(data)
      , nrows_(nrows)
      , ncols_(ncols) {}

  [[nodiscard]] __host__ __device__ std::size_t size() const {
    return nrows_ * ncols_;
  }

  [[nodiscard]] __host__ __device__ std::size_t nrows() const {
    return nrows_;
  }

  [[nodiscard]] __host__ __device__ std::size_t ncols() const {
    return ncols_;
  }

  __host__ __device__ AtomT& operator[](std::size_t n) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    return data_[n];
  }

  __host__ __device__ const AtomT& operator[](std::size_t n) const {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    return data_[n];
  }

  __host__ __device__ AtomT& operator()(std::size_t i, std::size_t j) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    return data_[i * ncols_ + j];
  }

  __host__ __device__ const AtomT& operator()(std::size_t i, std::size_t j) const {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    return data_[i * ncols_ + j];
  }
};

}  // namespace hsys
