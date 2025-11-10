#include <fill_rand.cuh>
#include <iostream>
#include <vadd/vadd.cuh>
#include <vadd/vadd_persist.cuh>
#include <vector>

using Vector = hsys::Vector<float>;

struct VAddPipeline {
 private:
  Vector a_;
  Vector b_;
  Vector c_;

 public:
  VAddPipeline(std::size_t size)
      : a_(size)
      , b_(size)
      , c_(size) {
    fill_rand(a_);
    fill_rand(b_);
  }

  template <class ImplT>
  void run(ImplT impl) {
    impl(c_, a_, b_);

    std::vector<float> result;
    c_.data().copy_to_host(result.data());

    for (auto x : result) {
      std::cout << x << std::endl;
    }
  }
};

int main() {
  VAddPipeline vadd10{10};
  vadd10.run(hsys::vadd_persist<>);
}
