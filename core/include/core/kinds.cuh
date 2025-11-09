#pragma once

#include <concepts>
#include <cuda_fp16.h>

namespace hsys {

template <class T>
concept AtomK = std::floating_point<T> || std::integral<T> || std::same_as<T, half>;

template <class T>
concept VectorK = requires { typename T::hsys_vector_feature; };

template <class T>
concept VectorViewK = requires { typename T::hsys_vector_view_feature; };

template <class T>
concept DataK = requires { typename T::hsys_data_feature; };

template <class T>
concept MatrixViewK = requires { typename T::hsys_matrix_view_feature; };

template <class T>
concept ViewK = MatrixViewK<T> or VectorViewK<T>;

template <class T>
concept MatrixK = requires { typename T::hsys_matrix_feature; };

}  // namespace hsys
