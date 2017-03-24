R"(

// Keep in sync with def.h!
typedef double floatType;

// inspired by
// https://devblogs.nvidia.com/parallelforall/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/

__kernel void matvecKernelCRS(__global int *ptr, __global int *index,
                              __global floatType *value, __global floatType *x,
                              __global floatType *y, int N) {
  for (int i = get_global_id(0); i < N; i += get_global_size(0)) {
    floatType tmp = 0;
    for (int j = ptr[i]; j < ptr[i + 1]; j++) {
      tmp += value[j] * x[index[j]];
    }
    y[i] = tmp;
  }
}

__kernel void matvecKernelELL(__global int *length, __global int *index,
                              __global floatType *data, __global floatType *x,
                              __global floatType *y, int N) {
  for (int i = get_global_id(0); i < N; i += get_global_size(0)) {
    floatType tmp = 0;
    for (int j = 0; j < length[i]; j++) {
      int k = j * N + i;
      tmp += data[k] * x[index[k]];
    }
    y[i] = tmp;
  }
}

__kernel void axpyKernel(floatType a, __global floatType *x,
                         __global floatType *y, int N) {
  for (int i = get_global_id(0); i < N; i += get_global_size(0)) {
    y[i] = a * x[i] + y[i];
  }
}

__kernel void xpayKernel(__global floatType *x, floatType a,
                         __global floatType *y, int N) {
  for (int i = get_global_id(0); i < N; i += get_global_size(0)) {
    y[i] = x[i] + a * y[i];
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

__kernel void vectorDotKernel(__global floatType *a, __global floatType *b,
                              __global floatType *tmp,
                              __local floatType *scratch, int N) {
  floatType sum = 0;
  for (int i = get_global_id(0); i < N; i += get_global_size(0)) {
    sum += a[i] * b[i];
  }

  sum = localReduceSum(sum, scratch);

  if (get_local_id(0) == 0) {
    tmp[get_group_id(0)] = sum;
  }
}

__kernel void applyPreconditionerKernelJacobi(__global floatType *C,
                                              __global floatType *x,
                                              __global floatType *y, int N) {
  for (int i = get_global_id(0); i < N; i += get_global_size(0)) {
    y[i] = C[i] * x[i];
  }
}

)"
