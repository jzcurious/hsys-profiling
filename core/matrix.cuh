#pragma once

#include "kinds.cuh"
#include "matrix_view.cuh"

#include <memory>
#include <work1/data.cuh>

namespace hsys {

template <AtomK AtomT, MatrixOpsPolicyK OpsPolicyT>
struct Matrix {
  struct hsys_matrix_feature {};

 private:
  std::shared_ptr<Data<AtomT>> data_;
  MatrixView<AtomT> view_;

 public:
  using ops_policy_t = OpsPolicyT;

  Matrix(std::size_t nrows, std::size_t ncols)
      : data_(std::make_shared<Data<AtomT>>(nrows * ncols))
      , view_(data_->data(), nrows, ncols) {}

  [[nodiscard]] std::size_t size() const {
    return data_->size();
  }

  [[nodiscard]] std::size_t nrows() const {
    return view_.nrows();
  }

  [[nodiscard]] std::size_t ncols() const {
    return view_.ncols();
  }

  Data<AtomT>& data() {
    return *data_;
  }

  const Data<AtomT>& data() const {
    return *data_;
  }

  MatrixView<AtomT>& view() {
    return view_;
  }

  const MatrixView<AtomT>& view() const {
    return view_;
  }
};

}  // namespace hsys
