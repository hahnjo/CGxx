#include <cassert>
#include <iostream>
#include <memory>

#include "../CG.h"
#include "CGCUDABase.h"
#include "kernel.h"
#include "utils.h"

/// Class implementing parallel kernels with CUDA.
class CGCUDA : public CGCUDABase {
  struct MatrixCRSCUDA : MatrixCRS, MatrixCRSDataCUDA {
    MatrixCRSCUDA(const MatrixCOO &coo) : MatrixCRS(coo) {}
  };
  struct MatrixELLCUDA : MatrixELL, MatrixELLDataCUDA {
    MatrixELLCUDA(const MatrixCOO &coo) : MatrixELL(coo) {}
  };

  Device device;

  virtual void init(const char *matrixFile) override;

  virtual void convertToMatrixCRS() override {
    matrixCRS.reset(new MatrixCRSCUDA(*matrixCOO));
  }
  virtual void convertToMatrixELL() override {
    matrixELL.reset(new MatrixELLCUDA(*matrixCOO));
  }

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
  checkError(cudaSetDevice(0));

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
  case MatrixFormatCRS: {
    size_t ptrSize = sizeof(int) * (N + 1);
    size_t indexSize = sizeof(int) * nz;
    size_t valueSize = sizeof(floatType) * nz;

    checkedMalloc(&device.matrixCRS.ptr, ptrSize);
    checkedMalloc(&device.matrixCRS.index, indexSize);
    checkedMalloc(&device.matrixCRS.value, valueSize);

    checkedMemcpyToDevice(device.matrixCRS.ptr, matrixCRS->ptr, ptrSize);
    checkedMemcpyToDevice(device.matrixCRS.index, matrixCRS->index, indexSize);
    checkedMemcpyToDevice(device.matrixCRS.value, matrixCRS->value, valueSize);
    break;
  }
  case MatrixFormatELL: {
    int elements = matrixELL->elements;
    size_t lengthSize = sizeof(int) * N;
    size_t indexSize = sizeof(int) * elements;
    size_t dataSize = sizeof(floatType) * elements;

    checkedMalloc(&device.matrixELL.length, lengthSize);
    checkedMalloc(&device.matrixELL.index, indexSize);
    checkedMalloc(&device.matrixELL.data, dataSize);

    checkedMemcpyToDevice(device.matrixELL.length, matrixELL->length,
                          lengthSize);
    checkedMemcpyToDevice(device.matrixELL.index, matrixELL->index, indexSize);
    checkedMemcpyToDevice(device.matrixELL.data, matrixELL->data, dataSize);
    break;
  }
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

  checkedMalloc(&device.tmp, sizeof(floatType) * Device::MaxBlocks);
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
  case MatrixFormatCRS: {
    checkedFree(device.matrixCRS.ptr);
    checkedFree(device.matrixCRS.index);
    checkedFree(device.matrixCRS.value);
    break;
  }
  case MatrixFormatELL: {
    checkedFree(device.matrixELL.length);
    checkedFree(device.matrixELL.index);
    checkedFree(device.matrixELL.data);
    break;
  }
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
    matvecKernelCRS<<<device.blocksMatvec, Device::Threads>>>(
        device.matrixCRS.ptr, device.matrixCRS.index, device.matrixCRS.value, x,
        y, N);
    break;
  case MatrixFormatELL:
    matvecKernelELL<<<device.blocksMatvec, Device::Threads>>>(
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

  axpyKernelCUDA<<<device.blocks, Device::Threads>>>(a, x, y, N);
  checkLastError();
  checkedSynchronize();
}

void CGCUDA::xpayKernel(Vector _x, floatType a, Vector _y) {
  floatType *x = device.getVector(_x);
  floatType *y = device.getVector(_y);

  xpayKernelCUDA<<<device.blocks, Device::Threads>>>(x, a, y, N);
  checkLastError();
  checkedSynchronize();
}

floatType CGCUDA::vectorDotKernel(Vector _a, Vector _b) {
  floatType res = 0;
  floatType *a = device.getVector(_a);
  floatType *b = device.getVector(_b);

  // This is needed for warpReduceSum on __CUDA_ARCH__ < 350
  size_t sharedForVectorDot =
      max(Device::Threads, BlockReduction) * sizeof(floatType);
  size_t sharedForReduce =
      max(Device::MaxBlocks, BlockReduction) * sizeof(floatType);

  // https://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/
  vectorDotKernelCUDA<<<device.blocks, Device::Threads, sharedForVectorDot>>>(
      a, b, device.tmp, N);
  checkLastError();
  deviceReduceKernel<<<1, Device::MaxBlocks, sharedForReduce>>>(
      device.tmp, device.tmp, device.blocks);
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
    applyPreconditionerKernelJacobi<<<device.blocks, Device::Threads>>>(
        device.jacobi.C, x, y, N);
    break;
  default:
    assert(0 && "Invalid preconditioner!");
  }
  checkLastError();
  checkedSynchronize();
}

CG *CG::getInstance() { return new CGCUDA; }
