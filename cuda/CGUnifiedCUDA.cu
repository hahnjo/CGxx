// SPDX-License-Identifier:	GPL-3.0-or-later

#include <cassert>
#include <iostream>

#include "../CG.h"
#include "kernel.h"
#include "utils.h"

/// Class implementing parallel kernels with CUDA and making use of unified
/// memory introduced with CUDA 6.
class CGUnifiedCUDA : public CG {
  struct ManagedMatrixCRS : MatrixCRS {
    virtual void allocatePtr(int rows) override {
      checkedMallocManaged(&ptr, sizeof(int) * (rows + 1));
    }
    virtual void deallocatePtr() override { checkedFree(ptr); }
    virtual void allocateIndexAndValue(int values) override {
      checkedMallocManaged(&index, sizeof(int) * values);
      checkedMallocManaged(&value, sizeof(floatType) * values);
    }
    virtual void deallocateIndexAndValue() override {
      checkedFree(index);
      checkedFree(value);
    }
  };
  struct ManagedMatrixELL : MatrixELL {
    virtual void allocateLength(int rows) override {
      checkedMallocManaged(&length, sizeof(int) * rows);
    }
    virtual void deallocateLength() override { checkedFree(length); }
    virtual void allocateIndexAndData() override {
      checkedMallocManaged(&index, sizeof(int) * elements);
      checkedMallocManaged(&data, sizeof(floatType) * elements);
    }
    virtual void deallocateIndexAndData() override {
      checkedFree(index);
      checkedFree(data);
    }
  };
  struct ManagedJacobi : Jacobi {
    virtual void allocateC(int N) override {
      checkedMallocManaged(&C, sizeof(floatType) * N);
    }
    virtual void deallocateC() override { checkedFree(C); }
  };

  /// Number of blocks for all kernels except CG#matvec.
  int blocks;
  /// Number of blocks for CG#matvec.
  int blocksMatvec;

  /// Temporary memory for use in reduction of CG#vectorDot.
  floatType *tmp = nullptr;

  /// CG#VectorP
  floatType *p = nullptr;
  /// CG#VectorQ
  floatType *q = nullptr;
  /// CG#VectorR
  floatType *r = nullptr;
  /// CG#VectorZ
  floatType *z = nullptr;

  floatType *getVector(Vector v) {
    switch (v) {
    case VectorK:
      return k;
    case VectorX:
      return x;
    case VectorP:
      return p;
    case VectorQ:
      return q;
    case VectorR:
      return r;
    case VectorZ:
      return z;
    }
    assert(0 && "Invalid value of v!");
    return nullptr;
  }

  virtual bool supportsMatrixFormat(MatrixFormat format) override {
    return format == MatrixFormatCRS || format == MatrixFormatELL;
  }
  virtual bool supportsPreconditioner(Preconditioner preconditioner) override {
    return preconditioner == PreconditionerJacobi;
  }

  virtual void init(const char *matrixFile) override;

  virtual void allocateMatrixCRS() override {
    matrixCRS.reset(new ManagedMatrixCRS);
  }
  virtual void allocateMatrixELL() override {
    matrixELL.reset(new ManagedMatrixELL);
  }

  virtual void allocateJacobi() override { jacobi.reset(new ManagedJacobi); }

  virtual void allocateK() override {
    checkedMallocManaged(&k, sizeof(floatType) * N);
  }
  virtual void deallocateK() override { checkedFree(k); }
  virtual void allocateX() override {
    checkedMallocManaged(&x, sizeof(floatType) * N);
  }
  virtual void deallocateX() override { checkedFree(x); }

  virtual void cpy(Vector _dst, Vector _src) override;
  virtual void matvecKernel(Vector _x, Vector _y) override;
  virtual void axpyKernel(floatType a, Vector _x, Vector _y) override;
  virtual void xpayKernel(Vector _x, floatType a, Vector _y) override;
  virtual floatType vectorDotKernel(Vector _a, Vector _b) override;

  virtual void applyPreconditionerKernel(Vector _x, Vector _y) override;

  virtual void cleanup() override {
    CG::cleanup();

    checkedFree(p);
    checkedFree(q);
    checkedFree(r);
    if (preconditioner == PreconditionerJacobi) {
      checkedFree(z);
    }
    checkedFree(tmp);
  }

public:
  CGUnifiedCUDA() : CG(MatrixFormatELL, PreconditionerJacobi) {}
};

void CGUnifiedCUDA::init(const char *matrixFile) {
  // Set the device for initialization.
  checkedSetDevice(0);

  CG::init(matrixFile);

  checkedMallocManaged(&p, sizeof(floatType) * N);
  checkedMallocManaged(&q, sizeof(floatType) * N);
  checkedMallocManaged(&r, sizeof(floatType) * N);
  if (preconditioner != PreconditionerNone) {
    checkedMallocManaged(&z, sizeof(floatType) * N);
  }
  checkedMalloc(&tmp, sizeof(floatType) * MaxBlocks);

  getLaunchConfiguration(N, blocks, blocksMatvec);
}

void CGUnifiedCUDA::cpy(Vector _dst, Vector _src) {
  floatType *dst = getVector(_dst);
  floatType *src = getVector(_src);

  checkedMemcpy(dst, src, sizeof(floatType) * N, cudaMemcpyDeviceToDevice);
}

void CGUnifiedCUDA::matvecKernel(Vector _x, Vector _y) {
  floatType *x = getVector(_x);
  floatType *y = getVector(_y);

  switch (matrixFormat) {
  case MatrixFormatCRS:
    matvecKernelCRS<<<blocksMatvec, Threads>>>(matrixCRS->ptr, matrixCRS->index,
                                               matrixCRS->value, x, y, N);
    break;
  case MatrixFormatELL:
    matvecKernelELL<<<blocksMatvec, Threads>>>(
        matrixELL->length, matrixELL->index, matrixELL->data, x, y, N);
    break;
  default:
    assert(0 && "Invalid matrix format!");
  }
  checkLastError();
  checkedSynchronize();
}

void CGUnifiedCUDA::axpyKernel(floatType a, Vector _x, Vector _y) {
  floatType *x = getVector(_x);
  floatType *y = getVector(_y);

  axpyKernelCUDA<<<blocks, Threads>>>(a, x, y, N);
  checkLastError();
  checkedSynchronize();
}

void CGUnifiedCUDA::xpayKernel(Vector _x, floatType a, Vector _y) {
  floatType *x = getVector(_x);
  floatType *y = getVector(_y);

  xpayKernelCUDA<<<blocks, Threads>>>(x, a, y, N);
  checkLastError();
  checkedSynchronize();
}

floatType CGUnifiedCUDA::vectorDotKernel(Vector _a, Vector _b) {
  floatType res = 0;
  floatType *a = getVector(_a);
  floatType *b = getVector(_b);

  // This is needed for warpReduceSum on __CUDA_ARCH__ < 350
  size_t sharedForVectorDot = max(Threads, BlockReduction) * sizeof(floatType);
  size_t sharedForReduce = max(MaxBlocks, BlockReduction) * sizeof(floatType);

  // https://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/
  vectorDotKernelCUDA<<<blocks, Threads, sharedForVectorDot>>>(a, b, tmp, N);
  checkLastError();
  deviceReduceKernel<<<1, MaxBlocks, sharedForReduce>>>(tmp, tmp, blocks);
  checkLastError();

  checkedMemcpy(&res, tmp, sizeof(floatType), cudaMemcpyDeviceToHost);
  // The device is synchronized by the memory transfer.

  return res;
}

void CGUnifiedCUDA::applyPreconditionerKernel(Vector _x, Vector _y) {
  floatType *x = getVector(_x);
  floatType *y = getVector(_y);

  switch (preconditioner) {
  case PreconditionerJacobi:
    applyPreconditionerKernelJacobi<<<blocks, Threads>>>(jacobi->C, x, y, N);
    break;
  default:
    assert(0 && "Invalid preconditioner!");
  }
  checkLastError();
  checkedSynchronize();
}

CG *CG::getInstance() { return new CGUnifiedCUDA; }
