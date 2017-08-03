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
