/*
    Copyright (C) 2017  Jonas Hahnfeld

    This file is part of CGxx.

    CGxx is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    CGxx is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with CGxx.  If not, see <http://www.gnu.org/licenses/>. */

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
