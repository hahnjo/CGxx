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

#include <cassert>
#include <iostream>

#include "../CG.h"
#include "CGCUDABase.h"
#include "kernel.h"
#include "utils.h"

/// Class implementing parallel kernels with CUDA.
class CGCUDA : public CGCUDABase {
  Device device;

  virtual void init(const char *matrixFile) override;

  virtual void doTransferTo() override;
  virtual void doTransferFrom() override;

  virtual void cpy(Vector _dst, Vector _src) override;
  virtual void matvecKernel(Vector _x, Vector _y) override;
  virtual void axpyKernel(floatType a, Vector _x, Vector _y) override;
  virtual void xpayKernel(Vector _x, floatType a, Vector _y) override;
  virtual floatType vectorDotKernel(Vector _a, Vector _b) override;

  virtual void applyPreconditionerKernel(Vector _x, Vector _y) override;
};

void CGCUDA::init(const char *matrixFile) {
  // Set the device for initialization.
  checkedSetDevice(0);

  CG::init(matrixFile);

  device.calculateLaunchConfiguration(N);
}

void CGCUDA::doTransferTo() {
  // Allocate memory on the device and transfer necessary data.
  size_t vectorSize = sizeof(floatType) * N;
  checkedMalloc(&device.k, vectorSize);
  checkedMalloc(&device.x, vectorSize);
  checkedMemcpyToDevice(device.k, k, vectorSize);
  checkedMemcpyToDevice(device.x, x, vectorSize);

  checkedMalloc(&device.p, vectorSize);
  checkedMalloc(&device.q, vectorSize);
  checkedMalloc(&device.r, vectorSize);

  switch (matrixFormat) {
  case MatrixFormatCRS:
    allocateAndCopyMatrixDataCRS(N, *matrixCRS, device.matrixCRS);
    break;
  case MatrixFormatELL:
    allocateAndCopyMatrixDataELL(N, *matrixELL, device.matrixELL);
    break;
  default:
    assert(0 && "Invalid matrix format!");
  }
  if (preconditioner != PreconditionerNone) {
    checkedMalloc(&device.z, vectorSize);

    switch (preconditioner) {
    case PreconditionerJacobi:
      checkedMalloc(&device.jacobi.C, vectorSize);
      checkedMemcpyToDevice(device.jacobi.C, jacobi->C, vectorSize);
      break;
    default:
      assert(0 && "Invalid preconditioner!");
    }
  }

  checkedMalloc(&device.tmp, sizeof(floatType) * MaxBlocks);
  checkedSynchronize();
}

void CGCUDA::doTransferFrom() {
  // Copy back solution and free memory on the device.
  checkedMemcpy(x, device.x, sizeof(floatType) * N, cudaMemcpyDeviceToHost);

  checkedFree(device.k);
  checkedFree(device.x);

  checkedFree(device.p);
  checkedFree(device.q);
  checkedFree(device.r);

  switch (matrixFormat) {
  case MatrixFormatCRS:
    freeMatrixCRSDevice(device.matrixCRS);
    break;
  case MatrixFormatELL:
    freeMatrixELLDevice(device.matrixELL);
    break;
  default:
    assert(0 && "Invalid matrix format!");
  }
  if (preconditioner != PreconditionerNone) {
    checkedFree(device.z);

    switch (preconditioner) {
    case PreconditionerJacobi: {
      checkedFree(device.jacobi.C);
      break;
    }
    default:
      assert(0 && "Invalid preconditioner!");
    }
  }

  checkedFree(device.tmp);
}

void CGCUDA::cpy(Vector _dst, Vector _src) {
  floatType *dst = device.getVector(_dst);
  floatType *src = device.getVector(_src);

  checkedMemcpy(dst, src, sizeof(floatType) * N, cudaMemcpyDeviceToDevice);
}

void CGCUDA::matvecKernel(Vector _x, Vector _y) {
  floatType *x = device.getVector(_x);
  floatType *y = device.getVector(_y);

  switch (matrixFormat) {
  case MatrixFormatCRS:
    matvecKernelCRS<<<device.blocksMatvec, Threads>>>(
        device.matrixCRS.ptr, device.matrixCRS.index, device.matrixCRS.value, x,
        y, N);
    break;
  case MatrixFormatELL:
    matvecKernelELL<<<device.blocksMatvec, Threads>>>(
        device.matrixELL.length, device.matrixELL.index, device.matrixELL.data,
        x, y, N);
    break;
  default:
    assert(0 && "Invalid matrix format!");
  }
  checkLastError();
  checkedSynchronize();
}

void CGCUDA::axpyKernel(floatType a, Vector _x, Vector _y) {
  floatType *x = device.getVector(_x);
  floatType *y = device.getVector(_y);

  axpyKernelCUDA<<<device.blocks, Threads>>>(a, x, y, N);
  checkLastError();
  checkedSynchronize();
}

void CGCUDA::xpayKernel(Vector _x, floatType a, Vector _y) {
  floatType *x = device.getVector(_x);
  floatType *y = device.getVector(_y);

  xpayKernelCUDA<<<device.blocks, Threads>>>(x, a, y, N);
  checkLastError();
  checkedSynchronize();
}

floatType CGCUDA::vectorDotKernel(Vector _a, Vector _b) {
  floatType res = 0;
  floatType *a = device.getVector(_a);
  floatType *b = device.getVector(_b);

  // This is needed for warpReduceSum on __CUDA_ARCH__ < 350
  size_t sharedForVectorDot = max(Threads, BlockReduction) * sizeof(floatType);
  size_t sharedForReduce = max(MaxBlocks, BlockReduction) * sizeof(floatType);

  // https://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/
  vectorDotKernelCUDA<<<device.blocks, Threads, sharedForVectorDot>>>(
      a, b, device.tmp, N);
  checkLastError();
  deviceReduceKernel<<<1, MaxBlocks, sharedForReduce>>>(device.tmp, device.tmp,
                                                        device.blocks);
  checkLastError();

  checkedMemcpy(&res, device.tmp, sizeof(floatType), cudaMemcpyDeviceToHost);
  // The device is synchronized by the memory transfer.

  return res;
}

void CGCUDA::applyPreconditionerKernel(Vector _x, Vector _y) {
  floatType *x = device.getVector(_x);
  floatType *y = device.getVector(_y);

  switch (preconditioner) {
  case PreconditionerJacobi:
    applyPreconditionerKernelJacobi<<<device.blocks, Threads>>>(device.jacobi.C,
                                                                x, y, N);
    break;
  default:
    assert(0 && "Invalid preconditioner!");
  }
  checkLastError();
  checkedSynchronize();
}

CG *CG::getInstance() { return new CGCUDA; }
