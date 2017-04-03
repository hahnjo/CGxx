#ifndef KERNEL_H
#define KERNEL_H

#include "../def.h"

const int BlockReduction = 32;

extern __global__ void matvecKernelCRS(int *ptr, int *index, floatType *value,
                                       floatType *x, floatType *y, int N);
extern __global__ void matvecKernelELL(int *length, int *index, floatType *data,
                                       floatType *x, floatType *y, int N);

extern __global__ void matvecKernelCRSRoundup(int *ptr, int *index,
                                              floatType *value, floatType *x,
                                              floatType *y, int N);
extern __global__ void matvecKernelELLRoundup(int *length, int *index,
                                              floatType *data, floatType *x,
                                              floatType *y, int N);

extern __global__ void axpyKernelCUDA(floatType a, floatType *x, floatType *y,
                                      int N);
extern __global__ void xpayKernelCUDA(floatType *x, floatType a, floatType *y,
                                      int N);

extern __global__ void deviceReduceKernel(floatType *in, floatType *out, int N);
extern __global__ void vectorDotKernelCUDA(floatType *a, floatType *b,
                                           floatType *tmp, int N);

extern __global__ void applyPreconditionerKernelJacobi(floatType *C,
                                                       floatType *x,
                                                       floatType *y, int N);

#endif
