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

#include "kernel.h"

// https://devblogs.nvidia.com/parallelforall/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/

template <bool roundup>
__inline__ __device__ void _matvecKernelCRS(int *ptr, int *index,
                                            floatType *value, floatType *x,
                                            floatType *y, int N) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += blockDim.x * gridDim.x) {
    // Skip load and store if nothing to be done...
    if (!roundup || ptr[i] != ptr[i + 1]) {
      floatType tmp = (roundup ? y[i] : 0);
      for (int j = ptr[i]; j < ptr[i + 1]; j++) {
        tmp += value[j] * x[index[j]];
      }
      y[i] = tmp;
    }
  }
}

template <bool roundup>
__inline__ __device__ void _matvecKernelELL(int *length, int *index,
                                            floatType *data, floatType *x,
                                            floatType *y, int N) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += blockDim.x * gridDim.x) {
    // Skip load and store if nothing to be done...
    if (!roundup || length[i] > 0) {
      floatType tmp = (roundup ? y[i] : 0);
      for (int j = 0; j < length[i]; j++) {
        int k = j * N + i;
        tmp += data[k] * x[index[k]];
      }
      y[i] = tmp;
    }
  }
}

__global__ void matvecKernelCRS(int *ptr, int *index, floatType *value,
                                floatType *x, floatType *y, int N) {
  _matvecKernelCRS<false>(ptr, index, value, x, y, N);
}
__global__ void matvecKernelELL(int *length, int *index, floatType *data,
                                floatType *x, floatType *y, int N) {
  _matvecKernelELL<false>(length, index, data, x, y, N);
}

__global__ void matvecKernelCRSRoundup(int *ptr, int *index, floatType *value,
                                       floatType *x, floatType *y, int N) {
  _matvecKernelCRS<true>(ptr, index, value, x, y, N);
}
__global__ void matvecKernelELLRoundup(int *length, int *index, floatType *data,
                                       floatType *x, floatType *y, int N) {
  _matvecKernelELL<true>(length, index, data, x, y, N);
}

// -----------------------------------------------------------------------------

__global__ void axpyKernelCUDA(floatType a, floatType *x, floatType *y, int N) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += blockDim.x * gridDim.x) {
    y[i] = a * x[i] + y[i];
  }
}

__global__ void xpayKernelCUDA(floatType *x, floatType a, floatType *y, int N) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += blockDim.x * gridDim.x) {
    y[i] = x[i] + a * y[i];
  }
}

// based on
// https://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/
// -----------------------------------------------------------------------------
#if __CUDA_ARCH__ >= 350
__inline__ __device__ floatType warpReduceSum(floatType val) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    val += __shfl_down(val, offset);
  }
  return val;
}
#else
__inline__ __device__ floatType warpReduceSum(floatType val) {
  // This makes shared point to the beginning of the shared memory which
  // has max(threads, blockReduction) elements, ie is large enough.
  // Warning: This uses the same memory as blockReduceSum, which should
  //          be fine at the moment.
  extern __shared__ floatType shared[];
  int tid = threadIdx.x;
  int lane = tid % warpSize;

  shared[tid] = val;
  __syncthreads();

  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    if (lane < offset) {
      shared[tid] += shared[tid + offset];
    }
    __syncthreads();
  }

  return shared[tid];
}
#endif

__inline__ __device__ floatType blockReduceSum(floatType val) {
  // Shared mem for partial sums
  static __shared__ floatType shared[BlockReduction];
  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  // Each warp performs partial reduction.
  val = warpReduceSum(val);

  // Write reduced value to shared memory.
  if (lane == 0) {
    shared[wid] = val;
  }

  // Wait for all partial reductions.
  __syncthreads();

  // Read from shared memory only if that warp existed.
  val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

  // Final reduce within first warp.
  if (wid == 0) {
    val = warpReduceSum(val);
  }

  return val;
}

__global__ void deviceReduceKernel(floatType *in, floatType *out, int N) {
  floatType sum = 0;
  // Reduce multiple elements per thread.
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += blockDim.x * gridDim.x) {
    sum += in[i];
  }

  sum = blockReduceSum(sum);

  if (threadIdx.x == 0) {
    out[blockIdx.x] = sum;
  }
}
// -----------------------------------------------------------------------------

__global__ void vectorDotKernelCUDA(floatType *a, floatType *b, floatType *tmp,
                                    int N) {
  floatType sum = 0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += blockDim.x * gridDim.x) {
    sum += a[i] * b[i];
  }
  sum = blockReduceSum(sum);

  if (threadIdx.x == 0) {
    tmp[blockIdx.x] = sum;
  }
}

__global__ void applyPreconditionerKernelJacobi(floatType *C, floatType *x,
                                                floatType *y, int N) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += blockDim.x * gridDim.x) {
    y[i] = C[i] * x[i];
  }
}
