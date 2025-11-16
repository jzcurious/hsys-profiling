#pragma once

namespace hsys {

template <class AtomT>
struct Data {
  struct hsys_data_feature {};

 private:
  std::size_t size_;
  AtomT* data_;

 public:
  using atom_t = AtomT;

  Data(std::size_t size)
      : size_(size)
      , data_(nullptr) {
    cudaMalloc(&data_, size * sizeof(AtomT));
  }

  Data(const Data& other)
      : size_(other.size_)
      , data_(nullptr) {
    cudaMalloc(&data_, size_ * sizeof(AtomT));
    cudaMemcpy(data_, other.data_, size_ * sizeof(AtomT), cudaMemcpyDeviceToDevice);
  }

  Data(Data&& other) noexcept
      : size_(other.size_)
      , data_(other.data_) {
    other.data_ = nullptr;
  }

  Data& operator=(const Data& other) {
    if (this != &other) {
      if (data_) cudaFree(data_);
      size_ = other.size_;
      cudaMalloc(&data_, size_ * sizeof(AtomT));
      cudaMemcpy(data_, other.data_, size_ * sizeof(AtomT), cudaMemcpyDeviceToDevice);
    }
    return *this;
  }

  Data& operator=(Data&& other) noexcept {
    if (this != &other) {
      if (data_) cudaFree(data_);
      size_ = other.size_;
      data_ = other.data_;
      other.data_ = nullptr;
    }
    return *this;
  }

  AtomT* data() {
    return data_;
  }

  const AtomT* data() const {
    return data_;
  }

  [[nodiscard]] std::size_t size() const {
    return size_;
  }

  // clang-format off

  void copy_to(
      AtomT* ptr,
      std::size_t dst_to = 0,
      std::size_t src_from = 0,
      std::size_t size = 0) const
 {
    cudaMemcpy(
      ptr + dst_to,
      data_ + src_from,
      (size ? size : size_) * sizeof(AtomT),
      cudaMemcpyDefault
    );
  }

  void copy_from(
      AtomT* ptr,
      std::size_t dst_to = 0,
      std::size_t src_from = 0,
      std::size_t size = 0) const
 {
    cudaMemcpy(
      data_ + src_from,
      ptr + dst_to,
      (size ? size : size_) * sizeof(AtomT),
      cudaMemcpyDefault
    );
  }

  // clang-format on

  ~Data() {
    if (data_) cudaFree(data_);
  }
};

}  // namespace hsys
