#pragma once

#include <core/matrix.cuh>
#include <core/matrix_view.cuh>

namespace hsys::kernels {

template <unsigned tile_size, unsigned extend_coef, AtomK AtomT>
__global__ void matmul_extend_symn(
    MatrixView<AtomT> c, const MatrixView<AtomT> a, const MatrixView<AtomT> b) {

  __shared__ AtomT tile_a[extend_coef][tile_size][tile_size];  // NOLINT
  __shared__ AtomT tile_b[extend_coef][tile_size][tile_size];  // NOLINT
  AtomT acc[extend_coef][extend_coef]{};  // NOLINT

  unsigned tx = threadIdx.x;
  unsigned ty = threadIdx.y;

  std::size_t m = a.nrows();
  std::size_t n = b.ncols();
  std::size_t k = a.ncols();

  const std::size_t i = extend_coef * tile_size * blockIdx.y + threadIdx.y;
  const std::size_t j = extend_coef * tile_size * blockIdx.x + threadIdx.x;

  for (unsigned v = 0; v < k; v += tile_size) {
    for (unsigned u = 0; u < extend_coef; ++u) {
      auto ii = i + u * tile_size;
      auto jj = j + u * tile_size;

      tile_a[u][ty][tx] = (v + tx < k) and (ii < m) ? a(ii, v + tx) : AtomT{0};
      tile_b[u][ty][tx] = (v + ty < k) and (jj < n) ? b(v + ty, jj) : AtomT{0};
    }

    __syncthreads();

    for (unsigned y = 0; y < extend_coef; ++y)
      for (unsigned x = 0; x < extend_coef; ++x)
        for (unsigned t = 0; t < tile_size; ++t)
          acc[y][x] += tile_a[y][ty][t] * tile_b[x][t][tx];

    __syncthreads();
  }

  for (unsigned y = 0; y < extend_coef; ++y)
    for (unsigned x = 0; x < extend_coef; ++x) {
      auto ii = y * tile_size + i;
      auto jj = x * tile_size + j;
      if (ii < m and jj < n) c(ii, jj) = acc[y][x];
    }
}

}  // namespace hsys::kernels

namespace hsys {

template <unsigned tile_size = 16,
    unsigned extend_coef = 4,
    MatrixK MatrixT = Matrix<float>>
void matmul_extend_symn(MatrixT& c, const MatrixT& a, const MatrixT& b) {
  constexpr auto block = dim3(tile_size, tile_size);
  const auto grid = dim3(std::ceil(c.ncols() / (block.x * extend_coef)),
      std::ceil(static_cast<float>(c.nrows()) / (block.y * extend_coef)));
  hsys::kernels::matmul_extend_symn<tile_size, extend_coef>
      <<<grid, block>>>(c.view(), a.view(), b.view());
}

}  // namespace hsys
