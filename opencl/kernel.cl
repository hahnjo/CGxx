R"(

// Keep in sync with def.h!
typedef double floatType;
typedef double8 floatType8;

// inspired by
// https://devblogs.nvidia.com/parallelforall/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/

// vec_type_hint tells the compiler that the kernel was optimized manually using
// vector data types. This actually not true but is the only way I found to
// prevent implicit vectorization by means of packing work-items together.
// Without this transformation, the loop vectorizer can successfully optimize
// the inner loop which is needed for better performance.
__attribute__((vec_type_hint(floatType8)))
__kernel void matvecKernelCRS(__global int *ptr, __global int *index,
                              __global floatType *value, __global floatType *x,
                              __global floatType *y, int yOffset, int N) {
  for (int i = get_global_id(0); i < N; i += get_global_size(0)) {
    floatType tmp = 0;
    for (int j = ptr[i]; j < ptr[i + 1]; j++) {
      tmp += value[j] * x[index[j]];
    }
    y[yOffset + i] = tmp;
  }
}

// See above...
__attribute__((vec_type_hint(floatType8)))
__kernel void matvecKernelCRSRoundup(__global int *ptr, __global int *index,
                                     __global floatType *value,
                                     __global floatType *x,
                                     __global floatType *y, int yOffset,
                                     int N) {
  for (int i = get_global_id(0); i < N; i += get_global_size(0)) {
    // Skip load and store if nothing to be done...
    if (ptr[i] != ptr[i + 1]) {
      floatType tmp = y[yOffset + i];
      for (int j = ptr[i]; j < ptr[i + 1]; j++) {
        tmp += value[j] * x[index[j]];
      }
      y[yOffset + i] = tmp;
    }
  }
}

__kernel void matvecKernelELL(__global int *length, __global int *index,
                              __global floatType *data, __global floatType *x,
                              __global floatType *y, int yOffset, int N) {
  for (int i = get_global_id(0); i < N; i += get_global_size(0)) {
    floatType tmp = 0;
    for (int j = 0; j < length[i]; j++) {
      int k = j * N + i;
      tmp += data[k] * x[index[k]];
    }
    y[yOffset + i] = tmp;
  }
}

__kernel void matvecKernelELLRoundup(__global int *length, __global int *index,
                                     __global floatType *data,
                                     __global floatType *x,
                                     __global floatType *y, int yOffset,
                                     int N) {
  for (int i = get_global_id(0); i < N; i += get_global_size(0)) {
    if (length[i] > 0) {
      floatType tmp = y[yOffset + i];
      for (int j = 0; j < length[i]; j++) {
        int k = j * N + i;
        tmp += data[k] * x[index[k]];
      }
      y[yOffset + i] = tmp;
    }
  }
}

__kernel void axpyKernel(floatType a, __global floatType *x, int xOffset,
                         __global floatType *y, int yOffset, int N) {
  for (int i = get_global_id(0); i < N; i += get_global_size(0)) {
    y[yOffset + i] = a * x[xOffset + i] + y[yOffset + i];
  }
}

__kernel void xpayKernel(__global floatType *x, int xOffset, floatType a,
                         __global floatType *y, int yOffset, int N) {
  for (int i = get_global_id(0); i < N; i += get_global_size(0)) {
    y[yOffset + i] = x[xOffset + i] + a * y[yOffset + i];
  }
}

// based on
// http://developer.amd.com/resources/articles-whitepapers/opencl-optimization-case-study-simple-reductions/
// inspired by
// https://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/
// -----------------------------------------------------------------------------
static inline floatType localReduceSum(floatType val,
                                       __local floatType *scratch) {
  int local_index = get_local_id(0);

  scratch[local_index] = val;
  barrier(CLK_LOCAL_MEM_FENCE);

  for (int offset = get_local_size(0) / 2; offset > 0; offset /= 2) {
    if (local_index < offset) {
      scratch[local_index] += scratch[local_index + offset];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  return scratch[local_index];
}

__kernel void deviceReduceKernel(__global floatType *in,
                                 __global floatType *out,
                                 __local floatType *scratch, int N) {
  floatType sum = 0;
  for (int i = get_global_id(0); i < N; i += get_global_size(0)) {
    sum += in[i];
  }

  sum = localReduceSum(sum, scratch);

  if (get_local_id(0) == 0) {
    out[get_group_id(0)] = sum;
  }
}
// -----------------------------------------------------------------------------

__kernel void vectorDotKernel(__global floatType *a, int aOffset,
                              __global floatType *b, int bOffset,
                              __global floatType *tmp,
                              __local floatType *scratch, int N) {
  floatType sum = 0;
  for (int i = get_global_id(0); i < N; i += get_global_size(0)) {
    sum += a[aOffset + i] * b[bOffset + i];
  }

  sum = localReduceSum(sum, scratch);

  if (get_local_id(0) == 0) {
    tmp[get_group_id(0)] = sum;
  }
}

__kernel void applyPreconditionerKernelJacobi(__global floatType *C,
                                              __global floatType *x,
                                              int xOffset,
                                              __global floatType *y,
                                              int yOffset, int N) {
  for (int i = get_global_id(0); i < N; i += get_global_size(0)) {
    y[yOffset + i] = C[i] * x[xOffset + i];
  }
}

)"
