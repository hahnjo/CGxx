#include <cassert>
#include <iostream>
#include <memory>

#include "../CG.h"
#include "../Matrix.h"
#include "../Preconditioner.h"
#include "kernel.h"

/// Class implementing parallel kernels with CUDA.
class CGCUDA : public CG {
  const int Threads = 128;
  const int MaxBlocks = 1024;
  // 65536 seems to not work on the Pascal nodes.
  const int MaxBlocksMatvec = 65535;
  int blocks;
  int blocksMatvec;

  floatType *tmp;

  floatType *k_dev;
  floatType *x_dev;

  floatType *p_dev;
  floatType *q_dev;
  floatType *r_dev;
  floatType *z_dev;

  struct {
    int *ptr;
    int *index;
    floatType *value;
  } matrixCRS_dev;
  struct {
    int *length;
    int *index;
    floatType *data;
  } matrixELL_dev;
  struct {
    floatType *C;
  } jacobi_dev;

  floatType *getVector(Vector v) {
    switch (v) {
    case VectorK:
      return k_dev;
    case VectorX:
      return x_dev;
    case VectorP:
      return p_dev;
    case VectorQ:
      return q_dev;
    case VectorR:
      return r_dev;
    case VectorZ:
      return z_dev;
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

  int getBlocks(int maxBlocks);
  virtual void init(const char *matrixFile) override;

  virtual bool needsTransfer() override { return true; }
  virtual void doTransferTo() override;
  virtual void doTransferFrom() override;

  virtual void cpy(Vector _dst, Vector _src) override;
  virtual void matvecKernel(Vector _x, Vector _y) override;
  virtual void axpyKernel(floatType a, Vector _x, Vector _y) override;
  virtual void xpayKernel(Vector _x, floatType a, Vector _y) override;
  virtual floatType vectorDotKernel(Vector _a, Vector _b) override;

  virtual void applyPreconditionerKernel(Vector _x, Vector _y) override;

public:
  CGCUDA() : CG(MatrixFormatELL, PreconditionerJacobi) {}
};

static inline void checkError(cudaError_t error) {
  if (error != cudaSuccess) {
    std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
    std::exit(1);
  }
}

static inline void checkLastError() { checkError(cudaGetLastError()); }

static inline void checkedSynchronize() { checkError(cudaDeviceSynchronize()); }

static inline void checkedMalloc(void *devPtr, size_t size) {
  checkError(cudaMalloc((void **)devPtr, size));
}

static inline void checkedMemcpy(void *dst, const void *src, size_t count,
                                 enum cudaMemcpyKind kind) {
  checkError(cudaMemcpy(dst, src, count, kind));
}

static inline void checkedMemcpyToDevice(void *dst, const void *src,
                                         size_t count) {
  checkedMemcpy(dst, src, count, cudaMemcpyHostToDevice);
}

static inline void checkedFree(void *devPtr) { checkError(cudaFree(devPtr)); }

int CGCUDA::getBlocks(int maxBlocks) {
  int maxNeededBlocks = (N + Threads - 1) / Threads;
  int blocks = maxNeededBlocks, div = 2;

  // We have grid-stride loops so it should be better if all blocks receive
  // roughly the same amount of work.
  while (blocks > maxBlocks) {
    blocks = maxNeededBlocks / div;
    div++;
  }

  return blocks;
}

void CGCUDA::init(const char *matrixFile) {
  // Set the device for initialization.
  checkError(cudaSetDevice(0));

  CG::init(matrixFile);

  blocks = getBlocks(MaxBlocks);
  blocksMatvec = getBlocks(MaxBlocksMatvec);
}

void CGCUDA::doTransferTo() {
  // Allocate memory on the device and transfer necessary data.
  size_t vectorSize = sizeof(floatType) * N;
  checkedMalloc(&k_dev, vectorSize);
  checkedMalloc(&x_dev, vectorSize);
  checkedMemcpyToDevice(k_dev, k.get(), vectorSize);
  checkedMemcpyToDevice(x_dev, x.get(), vectorSize);

  checkedMalloc(&p_dev, vectorSize);
  checkedMalloc(&q_dev, vectorSize);
  checkedMalloc(&r_dev, vectorSize);

  switch (matrixFormat) {
  case MatrixFormatCRS: {
    size_t ptrSize = sizeof(int) * (N + 1);
    size_t indexSize = sizeof(int) * nz;
    size_t valueSize = sizeof(floatType) * nz;

    checkedMalloc(&matrixCRS_dev.ptr, ptrSize);
    checkedMalloc(&matrixCRS_dev.index, indexSize);
    checkedMalloc(&matrixCRS_dev.value, valueSize);

    checkedMemcpyToDevice(matrixCRS_dev.ptr, matrixCRS->ptr.get(), ptrSize);
    checkedMemcpyToDevice(matrixCRS_dev.index, matrixCRS->index.get(),
                          indexSize);
    checkedMemcpyToDevice(matrixCRS_dev.value, matrixCRS->value.get(),
                          valueSize);
    break;
  }
  case MatrixFormatELL: {
    int elements = matrixELL->elements;
    size_t lengthSize = sizeof(int) * N;
    size_t indexSize = sizeof(int) * elements;
    size_t dataSize = sizeof(floatType) * elements;

    checkedMalloc(&matrixELL_dev.length, lengthSize);
    checkedMalloc(&matrixELL_dev.index, indexSize);
    checkedMalloc(&matrixELL_dev.data, dataSize);

    checkedMemcpyToDevice(matrixELL_dev.length, matrixELL->length.get(),
                          lengthSize);
    checkedMemcpyToDevice(matrixELL_dev.index, matrixELL->index.get(),
                          indexSize);
    checkedMemcpyToDevice(matrixELL_dev.data, matrixELL->data.get(), dataSize);
    break;
  }
  default:
    assert(0 && "Invalid matrix format!");
  }
  if (preconditioner != PreconditionerNone) {
    checkedMalloc(&z_dev, vectorSize);

    switch (preconditioner) {
    case PreconditionerJacobi:
      checkedMalloc(&jacobi_dev.C, vectorSize);
      checkedMemcpyToDevice(jacobi_dev.C, jacobi->C.get(), vectorSize);
      break;
    default:
      assert(0 && "Invalid preconditioner!");
    }
  }

  checkedMalloc(&tmp, sizeof(floatType) * MaxBlocks);
}

void CGCUDA::doTransferFrom() {
  // Copy back solution and free memory on the device.
  checkedMemcpy(x.get(), x_dev, sizeof(floatType) * N, cudaMemcpyDeviceToHost);

  checkedFree(k_dev);
  checkedFree(x_dev);

  checkedFree(p_dev);
  checkedFree(q_dev);
  checkedFree(r_dev);

  switch (matrixFormat) {
  case MatrixFormatCRS: {
    checkedFree(matrixCRS_dev.ptr);
    checkedFree(matrixCRS_dev.index);
    checkedFree(matrixCRS_dev.value);
    break;
  }
  case MatrixFormatELL: {
    checkedFree(matrixELL_dev.length);
    checkedFree(matrixELL_dev.index);
    checkedFree(matrixELL_dev.data);
    break;
  }
  default:
    assert(0 && "Invalid matrix format!");
  }
  if (preconditioner != PreconditionerNone) {
    checkedFree(z_dev);

    switch (preconditioner) {
    case PreconditionerJacobi: {
      checkedFree(jacobi_dev.C);
      break;
    }
    default:
      assert(0 && "Invalid preconditioner!");
    }
  }

  checkedFree(tmp);
}

void CGCUDA::cpy(Vector _dst, Vector _src) {
  floatType *dst = getVector(_dst);
  floatType *src = getVector(_src);

  checkedMemcpy(dst, src, sizeof(floatType) * N, cudaMemcpyDeviceToDevice);
}

void CGCUDA::matvecKernel(Vector _x, Vector _y) {
  floatType *x = getVector(_x);
  floatType *y = getVector(_y);

  switch (matrixFormat) {
  case MatrixFormatCRS:
    matvecKernelCRS<<<blocksMatvec, Threads>>>(
        matrixCRS_dev.ptr, matrixCRS_dev.index, matrixCRS_dev.value, x, y, N);
    break;
  case MatrixFormatELL:
    matvecKernelELL<<<blocksMatvec, Threads>>>(
        matrixELL_dev.length, matrixELL_dev.index, matrixELL_dev.data, x, y, N);
    break;
  default:
    assert(0 && "Invalid matrix format!");
  }
  checkLastError();
  checkedSynchronize();
}

void CGCUDA::axpyKernel(floatType a, Vector _x, Vector _y) {
  floatType *x = getVector(_x);
  floatType *y = getVector(_y);

  axpyKernelCUDA<<<blocks, Threads>>>(a, x, y, N);
  checkLastError();
  checkedSynchronize();
}

void CGCUDA::xpayKernel(Vector _x, floatType a, Vector _y) {
  floatType *x = getVector(_x);
  floatType *y = getVector(_y);

  xpayKernelCUDA<<<blocks, Threads>>>(x, a, y, N);
  checkLastError();
  checkedSynchronize();
}

floatType CGCUDA::vectorDotKernel(Vector _a, Vector _b) {
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

void CGCUDA::applyPreconditionerKernel(Vector _x, Vector _y) {
  floatType *x = getVector(_x);
  floatType *y = getVector(_y);

  switch (preconditioner) {
  case PreconditionerJacobi:
    applyPreconditionerKernelJacobi<<<blocks, Threads>>>(jacobi_dev.C, x, y, N);
    break;
  default:
    assert(0 && "Invalid preconditioner!");
  }
  checkLastError();
  checkedSynchronize();
}

CG *CG::getInstance() { return new CGCUDA; }
