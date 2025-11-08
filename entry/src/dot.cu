#include <core/vector.cuh>
#include <dot/dot_atomic.cuh>
#include <dot/dot_coars.cuh>
#include <dot/dot_nlog.cuh>
#include <dot/dot_shmem.cuh>
#include <dot/dot_vect.cuh>
#include <iostream>

#include <fill_rand.cuh>

using Vector = hsys::Vector<float>;

struct DotPipeline {
 private:
  Vector a_;
  Vector b_;
  Vector c_;

 public:
  DotPipeline(std::size_t len)
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
  DotPipeline dot{102400000};

  dot.run(hsys::dot_atomic<>);

  dot.run(hsys::dot_coars<2>);
  dot.run(hsys::dot_coars<4>);
  dot.run(hsys::dot_coars<8>);
  dot.run(hsys::dot_coars<16>);
  dot.run(hsys::dot_coars<32>);
  dot.run(hsys::dot_coars<64>);
  dot.run(hsys::dot_coars<128>);
  dot.run(hsys::dot_coars<256>);
  dot.run(hsys::dot_coars<512>);
  dot.run(hsys::dot_coars<1024>);

  dot.run(hsys::dot_vect<4>);
  dot.run(hsys::dot_vect<8>);
  dot.run(hsys::dot_vect<16>);
  dot.run(hsys::dot_vect<32>);
  dot.run(hsys::dot_vect<64>);
  dot.run(hsys::dot_vect<128>);
  dot.run(hsys::dot_vect<256>);
  dot.run(hsys::dot_vect<512>);

  dot.run(hsys::dot_shmem<4>);
  dot.run(hsys::dot_shmem<8>);
  dot.run(hsys::dot_shmem<16>);
  dot.run(hsys::dot_shmem<32>);
  dot.run(hsys::dot_shmem<64>);
  dot.run(hsys::dot_shmem<128>);
  dot.run(hsys::dot_shmem<256>);
  dot.run(hsys::dot_shmem<512>);

  dot.run(hsys::dot_nlog<32>);
  dot.run(hsys::dot_nlog<64>);
  dot.run(hsys::dot_nlog<128>);
  dot.run(hsys::dot_nlog<256>);
  dot.run(hsys::dot_nlog<512>);
}
