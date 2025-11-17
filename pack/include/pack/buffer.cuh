#pragma once

#include "dynpack.cuh"

namespace hsys {

template <DynPackKind DynPackT, class CallbackT>
  requires std::is_invocable_v<CallbackT, DynPackT>
struct TaskBuffer {
  using common_task_t = typename DynPackT::common_task_t;

  TaskBuffer(DynPackT&& pack, const CallbackT& callback)
      : pack_(std::move(pack))
      , callback_(std::move(callback)) {}

  void flush() {
    callback_(pack_);
  }

  void push(const common_task_t& task) {
    if (pack_.is_full()) flush();
    pack_.push(task);
  }

 private:
  DynPackT pack_{};
  CallbackT callback_;
};

}  // namespace hsys
