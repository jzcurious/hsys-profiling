#include <core/vector.cuh>
#include <iostream>
#include <kernels/dot_atomic.cuh>

#include "../include/fill_rand.cuh"

using Vector = hsys::Vector<float>;

int main() {
  auto a = Vector(102400);
  auto b = Vector(102400);
  auto c = Vector(1);

  fill_rand(a);
  fill_rand(b);

  hsys::dot_atomic<<<800, 128>>>(c.view(), a.view(), b.view());

  float result = 0;
  c.data().copy_to_host(&result);
  std::cout << result << std::endl;
}
