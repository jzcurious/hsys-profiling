#pragma once

#include "data.cuh"
#include "vector_view.cuh"

#include <memory>

namespace hsys {

template <AtomK AtomT>
struct Vector {
  struct hsys_vector_feature {};

 private:
  std::shared_ptr<Data<AtomT>> data_;
  VectorView<AtomT> view_;

 public:
  Vector(std::size_t size)
      : data_(std::make_shared<Data<AtomT>>(size))
      , view_(data_->data(), size) {}

  [[nodiscard]] std::size_t size() const {
    return data_->size();
  }

  Data<AtomT>& data() {
    return *data_;
  }

  const Data<AtomT>& data() const {
    return *data_;
  }

  VectorView<AtomT>& view() {
    return view_;
  }

  const VectorView<AtomT>& view() const {
    return view_;
  }
};

}  // namespace hsys
