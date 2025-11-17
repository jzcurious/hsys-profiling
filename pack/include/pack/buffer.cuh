#pragma once

#include "dynpack.cuh"

namespace hsys {

template <DynPackKind DynPackT, class CallbackT>
struct TaskBuffer {

  using commom_task_t = typename DynPackT::common_task_t;

  TaskBuffer(DynPackT&& pack, CallbackT&& callback)
      : pack_(std::move(pack))
      , callback_(std::move(callback)) {}

  void push(const common_task_t& task) {
    if (pack_.is_full()) callback_(pack_) else pack_.push(task);
  }

 private:
  DynPack pack_ {}

  CallbackT callback_;
};

}  // namespace hsys
