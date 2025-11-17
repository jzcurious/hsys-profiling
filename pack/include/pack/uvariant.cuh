#pragma once

#include <cuda/std/variant>
#include <tuple>

namespace hsys {

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

}  // namespace hsys
