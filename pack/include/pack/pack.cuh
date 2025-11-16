#pragma once

#include <array>
#include <cuda/std/variant>
#include <task/task.cuh>
#include <tuple>

namespace hsys::internal {

template <typename... Ts>
struct make_unique_tuple;

template <typename T, typename... Ts>
struct make_unique_tuple<T, Ts...> {
  using type = decltype([] {
    if constexpr (sizeof...(Ts) == 0) {
      return std::tuple<T>{};
    } else {
      using RestTuple = typename make_unique_tuple<Ts...>::type;
      if constexpr ((std::is_same_v<T, Ts> or ...)) {
        return RestTuple{};
      } else {
        return std::tuple_cat(std::tuple<T>{}, RestTuple{});
      }
    }
  }());
};

template <>
struct make_unique_tuple<> {
  using type = std::tuple<>;
};

template <typename... Ts>
using make_unique_tuple_t = typename make_unique_tuple<Ts...>::type;

template <class T>
struct tuple_to_variant;

template <class... Ts>
struct tuple_to_variant<std::tuple<Ts...>> {
  using type = cuda::std::variant<Ts...>;
};

template <class... Ts>
using make_unique_variant_t =
    typename tuple_to_variant<typename make_unique_tuple<Ts...>::type>::type;

}  // namespace hsys::internal

namespace hsys {

template <class T>
concept PackKind = requires { typename T::hsys_pack_kind; };

template <unsigned size, class TaskVarT>
struct Pack {
  struct hsys_pack_kind {};

  using common_task_t = TaskVarT;
  using tasks_t = std::array<common_task_t, size>;

  Pack(const Pack& other) = default;
  Pack(Pack&& other) = default;

  Pack& operator=(const Pack& other) = default;
  Pack& operator=(Pack&& other) = default;

  template <TaskKind... T>
  __host__ __device__ Pack(const T&... task)
      : tasks_{task...} {}

  __host__ __device__ tasks_t& tasks() {
    return tasks_;
  }

  ~Pack() = default;

 private:
  tasks_t tasks_;
};

template <class... T>
__host__ __device__ Pack(const T&... task)
    -> Pack<sizeof...(T), internal::make_unique_variant_t<T...>>;

}  // namespace hsys
