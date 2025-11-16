#pragma once

#include "pack.cuh"
#include <core/data.cuh>

namespace hsys {

template <class TaskVarT>
struct DynPack {
  struct hsys_dynpack_kind {};

  using common_task_t = TaskVarT;

  DynPack(std::size_t capacity)
      : data_(capacity) {}

  [[nodiscard]] std::size_t capacity() const {
    return data_.size();
  }

  void push(const TaskVarT& task) {
    if (capacity() > size_) {

    } else {
    }
    ++size_;
  }

 private:
  Data<TaskVarT> data_;
  std::size_t size_ = 0;
};

}  // namespace hsys
