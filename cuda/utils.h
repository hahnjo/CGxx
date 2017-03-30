#ifndef UTILS_H
#define UTILS_H

#include <cmath>

static inline void checkError(cudaError_t error) {
  if (error != cudaSuccess) {
    std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
    std::exit(1);
  }
}

static inline void checkLastError() { checkError(cudaGetLastError()); }

static inline void checkedSetDevice(int device) {
  checkError(cudaSetDevice(device));
}

static inline void checkedSynchronize() { checkError(cudaDeviceSynchronize()); }

static inline void checkedMalloc(void *devPtr, size_t size) {
  checkError(cudaMalloc((void **)devPtr, size));
}

static inline void checkedMemcpy(void *dst, const void *src, size_t count,
                                 enum cudaMemcpyKind kind) {
  checkError(cudaMemcpy(dst, src, count, kind));
}

static inline void checkedMemcpyAsync(void *dst, const void *src, size_t count,
                                      enum cudaMemcpyKind kind,
                                      cudaStream_t stream = 0) {
  checkError(cudaMemcpyAsync(dst, src, count, kind, stream));
}

static inline void checkedMemcpyToDevice(void *dst, const void *src,
                                         size_t count) {
  checkedMemcpy(dst, src, count, cudaMemcpyHostToDevice);
}

static inline void checkedMemcpyAsyncToDevice(void *dst, const void *src,
                                              size_t count,
                                              cudaStream_t stream = 0) {
  checkedMemcpyAsync(dst, src, count, cudaMemcpyHostToDevice, stream);
}

static inline void checkedFree(void *devPtr) { checkError(cudaFree(devPtr)); }

static inline void checkedMallocHost(void *ptr, size_t size) {
  checkError(cudaMallocHost((void **)ptr, size));
}

static inline void checkedFreeHost(void *ptr) {
  checkError(cudaFreeHost(ptr));
}

static inline int calculateBlocks(int N, int threads, int maxBlocks) {
  int blocks, div = 1;

  // We have grid-stride loops so it should be better if all blocks receive
  // roughly the same amount of work.
  do {
    blocks = std::ceil(((double)N) / threads / div);
    div++;
  } while (blocks > maxBlocks);

  return blocks;
}

#endif
