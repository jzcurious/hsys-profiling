#include <core/vector.cuh>
#include <iostream>
#include <kernels/dot_atomic.cuh>
#include <kernels/dot_coarsened.cuh>
#include <kernels/dot_nlog.cuh>
#include <kernels/dot_shmem.cuh>
#include <kernels/dot_vectorized.cuh>

#include "../include/fill_rand.cuh"

using Vector = hsys::Vector<float>;

void dot_atomic(Vector& c, const Vector& a, const Vector& b) {
  constexpr unsigned block = 128;
  const unsigned grid = std::ceil(a.size() / block);
  hsys::dot_atomic<<<grid, block>>>(c.view(), a.view(), b.view());
}

template <unsigned portion = 32>
void dot_coarsened(Vector& c, const Vector& a, const Vector& b) {
  constexpr unsigned block = 32;
  const unsigned grid = std::ceil(a.size() / (block * portion));
  hsys::dot_coarsened<portion><<<grid, block>>>(c.view(), a.view(), b.view());
}

template <unsigned portion = 32>
void dot_vectorized(Vector& c, const Vector& a, const Vector& b) {
  constexpr unsigned block = 32;
  const unsigned grid = std::ceil(a.size() / (block * portion));
  hsys::dot_vectorized<portion><<<grid, block>>>(c.view(), a.view(), b.view());
}

template <unsigned portion = 32>
void dot_shmem(Vector& c, const Vector& a, const Vector& b) {
  constexpr unsigned block = 32;
  const unsigned grid = std::ceil(a.size() / (block * portion));
  hsys::dot_shmem<portion><<<grid, block>>>(c.view(), a.view(), b.view());
}

template <unsigned block = 32>
void dot_nlog(Vector& c, const Vector& a, const Vector& b) {
  const unsigned grid = std::ceil(a.size() / block);
  hsys::dot_nlog<block><<<grid, block>>>(c.view(), a.view(), b.view());
}

struct DotPipeline {
 private:
  Vector a_;
  Vector b_;
  Vector c_;

 public:
  DotPipeline(std::size_t len = 102400)
      : a_(len)
      , b_(len)
      , c_(1) {
    fill_rand(a_);
    fill_rand(b_);
  }

  template <class ImplT>
  void run(ImplT impl) {
    float result = 0;
    impl(c_, a_, b_);
    c_.data().copy_to_host(&result);
    std::cout << result << std::endl;
  }
};

int main() {
  DotPipeline dot;

  dot.run(dot_atomic);

  dot.run(dot_coarsened<2>);
  dot.run(dot_coarsened<4>);
  dot.run(dot_coarsened<8>);
  dot.run(dot_coarsened<16>);
  dot.run(dot_coarsened<32>);
  dot.run(dot_coarsened<64>);
  dot.run(dot_coarsened<128>);
  dot.run(dot_coarsened<256>);
  dot.run(dot_coarsened<512>);
  dot.run(dot_coarsened<1024>);

  dot.run(dot_vectorized<4>);
  dot.run(dot_vectorized<8>);
  dot.run(dot_vectorized<16>);
  dot.run(dot_vectorized<32>);
  dot.run(dot_vectorized<64>);
  dot.run(dot_vectorized<128>);
  dot.run(dot_vectorized<256>);
  dot.run(dot_vectorized<512>);

  dot.run(dot_shmem<4>);
  dot.run(dot_shmem<8>);
  dot.run(dot_shmem<16>);
  dot.run(dot_shmem<32>);
  dot.run(dot_shmem<64>);
  dot.run(dot_shmem<128>);
  dot.run(dot_shmem<256>);
  dot.run(dot_shmem<512>);

  dot.run(dot_nlog<32>);
  dot.run(dot_nlog<64>);
  dot.run(dot_nlog<128>);
  dot.run(dot_nlog<256>);
  dot.run(dot_nlog<512>);
}
