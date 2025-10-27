#include <core/vector.cuh>
#include <iostream>
#include <kernels/dot_atomic.cuh>
#include <kernels/dot_coarsened.cuh>

#include "../include/fill_rand.cuh"

using Vector = hsys::Vector<float>;

void run_dot_atomic(Vector& c, const Vector& a, const Vector& b) {
  constexpr unsigned block = 128;
  const unsigned grid = std::ceil(a.size() / block);
  hsys::dot_atomic<<<grid, block>>>(c.view(), a.view(), b.view());
}

template <unsigned portion = 32>
void run_dot_coarsened(Vector& c, const Vector& a, const Vector& b) {
  const unsigned block = std::ceil(128 / portion);
  const unsigned grid = std::ceil(a.size() / block);
  hsys::dot_coarsened<portion><<<grid, block>>>(c.view(), a.view(), b.view());
}

template <class ImplT>
void dot_n(ImplT impl, std::size_t n = 102400) {
  auto a = Vector(n);
  auto b = Vector(n);
  auto c = Vector(1);

  fill_rand(a);
  fill_rand(b);

  impl(c, a, b);

  float result = 0;
  c.data().copy_to_host(&result);
  std::cout << result << std::endl;
}

int main() {
  dot_n(run_dot_atomic);
  dot_n(run_dot_coarsened<8>);
  dot_n(run_dot_coarsened<16>);
  dot_n(run_dot_coarsened<32>);
  dot_n(run_dot_coarsened<64>);
  dot_n(run_dot_coarsened<128>);
}
