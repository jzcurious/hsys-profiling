#include <core/matrix.cuh>
#include <matmul/matmul.cuh>
#include <matmul/matmul_extend_adown.cuh>
#include <matmul/matmul_extend_bright.cuh>
#include <matmul/matmul_extend_sym2.cuh>
#include <matmul/matmul_extend_sym3.cuh>
#include <matmul/matmul_extend_symn.cuh>

#include "../include/fill_rand.cuh"

using Matrix = hsys::Matrix<float>;

struct MatmulPipeline {
 private:
  Matrix a_;
  Matrix b_;
  Matrix c_;

 public:
  MatmulPipeline(std::size_t size)
      : a_(size, size)
      , b_(size, size)
      , c_(size, size) {
    fill_rand(a_);
    fill_rand(b_);
  }

  template <class ImplT>
  void run(ImplT impl) {
    impl(c_, a_, b_);
  }
};

int main() {
  MatmulPipeline matmul{1024};

  matmul.run(hsys::matmul<>);
  matmul.run(hsys::matmul_extend_bright<>);
  matmul.run(hsys::matmul_extend_adown<>);
  matmul.run(hsys::matmul_extend_sym2<>);
  matmul.run(hsys::matmul_extend_sym3<>);
  matmul.run(hsys::matmul_extend_symn<>);
}
