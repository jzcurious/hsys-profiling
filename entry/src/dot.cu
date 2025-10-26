#include <core/vector.cuh>
#include <iostream>
#include <kernels/dot_atomic.cuh>

#include "../include/fill_rand.cuh"

using Vector = hsys::Vector<float>;

int main() {
  auto a = Vector(1024);
  auto b = Vector(1024);
  auto c = Vector(1);

  fill_rand(a);
  fill_rand(b);

  hsys::dot_atomic<<<8, 128>>>(c.view(), a.view(), b.view());

  float result = 0;
  c.data().copy_to_host(&result);
  std::cout << result << std::endl;
}
