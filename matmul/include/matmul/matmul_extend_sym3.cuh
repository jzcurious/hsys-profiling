#pragma once

#include <core/matrix.cuh>
#include <core/matrix_view.cuh>

namespace hsys::kernels {

template <unsigned tile_size, AtomK AtomT>
__global__ void matmul_extend_sym3(
    MatrixView<AtomT> c, const MatrixView<AtomT> a, const MatrixView<AtomT> b) {

  __shared__ AtomT tile_a1[tile_size][tile_size];  // NOLINT
  __shared__ AtomT tile_b1[tile_size][tile_size];  // NOLINT

  __shared__ AtomT tile_a2[tile_size][tile_size];  // NOLINT
  __shared__ AtomT tile_b2[tile_size][tile_size];  // NOLINT

  __shared__ AtomT tile_a3[tile_size][tile_size];  // NOLINT
  __shared__ AtomT tile_b3[tile_size][tile_size];  // NOLINT

  std::size_t tx = threadIdx.x;
  std::size_t ty = threadIdx.y;

  if (tx >= tile_size) return;
  if (ty >= tile_size) return;

  const std::size_t i = 3 * tile_size * blockIdx.y + threadIdx.y;
  const std::size_t j = 3 * tile_size * blockIdx.x + threadIdx.x;

  std::size_t m = a.nrows();
  std::size_t n = b.ncols();
  std::size_t k = a.ncols();

  AtomT acc1{0}, acc2{0}, acc3{0}, acc4{0}, acc5{0}, acc6{0}, acc7{0}, acc8{0}, acc9{0};

  for (std::size_t v = 0; v < k; v += tile_size) {
    tile_a1[ty][tx] = (v + tx < k) and (i < m) ? a(i, v + tx) : AtomT{0};
    tile_b1[ty][tx] = (v + ty < k) and (j < n) ? b(v + ty, j) : AtomT{0};
    tile_a2[ty][tx]
        = (v + tx < k) and (i + tile_size < m) ? a(i + tile_size, v + tx) : AtomT{0};
    tile_b2[ty][tx]
        = (v + ty < k) and (j + tile_size < n) ? b(v + ty, j + tile_size) : AtomT{0};
    tile_a3[ty][tx] = (v + tx < k) and (i + 2 * tile_size < m)
                          ? a(i + 2 * tile_size, v + tx)
                          : AtomT{0};
    tile_b3[ty][tx] = (v + ty < k) and (j + 2 * tile_size < n)
                          ? b(v + ty, j + 2 * tile_size)
                          : AtomT{0};

    __syncthreads();

    for (std::size_t t = 0; t < tile_size; ++t) {
      acc1 += tile_a1[ty][t] * tile_b1[t][tx];
      acc2 += tile_a1[ty][t] * tile_b2[t][tx];
      acc3 += tile_a1[ty][t] * tile_b3[t][tx];

      acc4 += tile_a2[ty][t] * tile_b1[t][tx];
      acc5 += tile_a2[ty][t] * tile_b2[t][tx];
      acc6 += tile_a2[ty][t] * tile_b3[t][tx];

      acc7 += tile_a3[ty][t] * tile_b1[t][tx];
      acc8 += tile_a3[ty][t] * tile_b2[t][tx];
      acc9 += tile_a3[ty][t] * tile_b3[t][tx];
    }

    __syncthreads();
  }

  if (i < m and j < n) c(i, j) = acc1;
  if (i < m and j + tile_size < n) c(i, j + tile_size) = acc2;
  if (i < m and j + 2 * tile_size < n) c(i, j + 2 * tile_size) = acc3;

  if (i + tile_size < m and j < n) c(i + tile_size, j) = acc4;
  if (i + tile_size < m and j + tile_size < n) c(i + tile_size, j + tile_size) = acc5;
  if (i + tile_size < m and j + 2 * tile_size < n)
    c(i + tile_size, j + 2 * tile_size) = acc6;

  if (i + 2 * tile_size < m and j < n) c(i + 2 * tile_size, j) = acc7;
  if (i + 2 * tile_size < m and j + tile_size < n)
    c(i + 2 * tile_size, j + tile_size) = acc8;
  if (i + 2 * tile_size < m and j + 2 * tile_size < n)
    c(i + 2 * tile_size, j + 2 * tile_size) = acc9;
}

}  // namespace hsys::kernels

namespace hsys {

template <MatrixK MatrixT = Matrix<float>>
void matmul_extend_sym3(MatrixT& c, const MatrixT& a, const MatrixT& b) {
  constexpr auto block = dim3(16, 16);
  const auto grid
      = dim3(std::ceil(c.ncols() / (block.x * 2)), std::ceil(c.nrows() / (block.y * 2)));
  hsys::kernels::matmul_extend_sym3<16><<<grid, block>>>(c.view(), a.view(), b.view());
}

}  // namespace hsys
