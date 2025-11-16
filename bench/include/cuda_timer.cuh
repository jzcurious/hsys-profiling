#pragma once

struct CUDATimer {
  CUDATimer(float* elapse_time_s)
      : elapse_time_s_(elapse_time_s) {
    cudaEventCreate(&start_);
    cudaEventCreate(&stop_);
    cudaEventRecord(start_);
  }

  CUDATimer(const CUDATimer&) = delete;
  CUDATimer& operator=(const CUDATimer&) = delete;

  CUDATimer(CUDATimer&&) = delete;
  CUDATimer& operator=(CUDATimer&&) = delete;

  ~CUDATimer() {
    cudaEventRecord(stop_);
    cudaEventSynchronize(stop_);
    cudaEventElapsedTime(elapse_time_s_, start_, stop_);
    *elapse_time_s_ /= 1000;
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }

 private:
  float* elapse_time_s_;
  cudaEvent_t start_ = nullptr;
  cudaEvent_t stop_ = nullptr;
};
