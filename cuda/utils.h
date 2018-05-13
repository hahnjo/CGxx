// SPDX-License-Identifier:	GPL-3.0-or-later

#ifndef UTILS_H
#define UTILS_H

#include <cmath>
#include <cstdlib>
#include <iostream>

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

static inline void checkedMallocManaged(void *devPtr, size_t size) {
  checkError(cudaMallocManaged((void **)devPtr, size));
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

/// Number of threads for all kernels.
static const int Threads = 128;
/// Maximum number of blocks for all kernels except CG#matvec.
static const int MaxBlocks = 1024;
/// Maximum number of blocks for CG#matvec.
/// (65536 seems to not work on the Pascal nodes!)
static const int MaxBlocksMatvec = 65535;

/// Calculate the launch configuration for vectors of length \a N and store
/// the parameters in \a blocks and \a blocksMatvec.
static inline void getLaunchConfiguration(int N, int &blocks, int &blocksMatvec) {
  blocks = calculateBlocks(N, Threads, MaxBlocks);
  blocksMatvec = calculateBlocks(N, Threads, MaxBlocksMatvec);
}

#endif
