#pragma once

#include <core/matrix.cuh>
#include <core/matrix_view.cuh>

namespace hsys::kernels {

template <std::size_t tile_size, AtomK AtomT>
__global__ void matmul(
    MatrixView<AtomT> c, const MatrixView<AtomT> a, const MatrixView<AtomT> b) {

  __shared__ AtomT tile_a[tile_size][tile_size];  // NOLINT
  __shared__ AtomT tile_b[tile_size][tile_size];  // NOLINT

  const std::size_t i = tile_size * blockIdx.y + threadIdx.y;
  const std::size_t j = tile_size * blockIdx.x + threadIdx.x;

  std::size_t tx = threadIdx.x;
  std::size_t ty = threadIdx.y;

  std::size_t m = a.nrows();
  std::size_t n = b.ncols();
  std::size_t k = a.ncols();

  AtomT acc{0};

  for (std::size_t v = 0; v < k; v += tile_size) {
    tile_a[ty][tx] = (v + tx < k) and (i < m) ? a(i, v + tx) : AtomT{0};
    tile_b[ty][tx] = (v + ty < k) and (j < n) ? b(v + ty, j) : AtomT{0};

    __syncthreads();

    for (std::size_t t = 0; t < tile_size; ++t) {
      acc += tile_a[ty][t] * tile_b[t][tx];
    }

    __syncthreads();
  }

  if (i < m and j < n) c(i, j) = acc;
}

}  // namespace hsys::kernels

namespace hsys {

template <MatrixK MatrixT = Matrix<float>>
void matmul(MatrixT& c, const MatrixT& a, const MatrixT& b) {
  constexpr auto block = dim3(16, 16);
  const auto grid = dim3(
      std::ceil(c.ncols() / block.x), std::ceil(static_cast<float>(c.nrows()) / block.y));
  hsys::kernels::matmul<16><<<grid, block>>>(c.view(), a.view(), b.view());
}

}  // namespace hsys
