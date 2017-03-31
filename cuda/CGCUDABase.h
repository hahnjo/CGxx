#ifndef CG_CUDA_BASE_H
#define CG_CUDA_BASE_H

#include <cassert>

#include "../CG.h"
#include "../Matrix.h"
#include "../Preconditioner.h"
#include "utils.h"

/// Class implementing parallel kernels with CUDA.
class CGCUDABase : public CG {
protected:
  /// Derived struct for pinned memory.
  struct MatrixDataCRSCUDA : MatrixDataCRS {
    virtual void allocatePtr(int rows) override {
      checkedMallocHost(&ptr, sizeof(int) * (rows + 1));
    }
    virtual void deallocatePtr() override { checkedFreeHost(ptr); }
    virtual void allocateIndexAndValue(int values) override {
      checkedMallocHost(&index, sizeof(int) * values);
      checkedMallocHost(&value, sizeof(floatType) * values);
    }
    virtual void deallocateIndexAndValue() override {
      checkedFreeHost(index);
      checkedFreeHost(value);
    }
  };
  /// Derived struct for pinned memory.
  struct MatrixDataELLCUDA : MatrixDataELL {
    virtual void allocateLength(int rows) override {
      checkedMallocHost(&length, sizeof(int) * rows);
    }
    virtual void deallocateLength() override { checkedFreeHost(length); }
    virtual void allocateIndexAndData() override {
      checkedMallocHost(&index, sizeof(int) * elements);
      checkedMallocHost(&data, sizeof(floatType) * elements);
    }
    virtual void deallocateIndexAndData() override {
      checkedFreeHost(index);
      checkedFreeHost(data);
    }
  };
  /// Derived struct for pinned memory.
  struct JacobiCUDA : Jacobi {
    /// @see Jacobi
    JacobiCUDA(const MatrixCOO &coo) : Jacobi(coo) {}

    virtual void allocateC(int N) override {
      checkedMallocHost(&C, sizeof(floatType) * N);
    }
    virtual void deallocateC() override { checkedFreeHost(C); }
  };

  /// Holds information about a single device, especially memory and its
  /// launch configuration.
  struct Device {
    /// Number of threads for all kernels.
    static const int Threads = 128;
    /// Maximum number of blocks for all kernels except CG#matvec.
    static const int MaxBlocks = 1024;
    /// Maximum number of blocks for CG#matvec.
    /// (65536 seems to not work on the Pascal nodes!)
    static const int MaxBlocksMatvec = 65535;

    /// Number of blocks for all kernels except CG#matvec.
    int blocks;
    /// Number of blocks for CG#matvec.
    int blocksMatvec;

    /// Temporary memory for use in reduction of CG#vectorDot.
    floatType *tmp;

    /// CG#VectorK
    floatType *k;
    /// CG#VectorX
    floatType *x;

    /// CG#VectorP
    floatType *p;
    /// CG#VectorQ
    floatType *q;
    /// CG#VectorR
    floatType *r;
    /// CG#VectorZ
    floatType *z;

    /// MatrixCRS or SplitMatrixCRS on the device.
    struct {
      int *ptr;
      int *index;
      floatType *value;
    } matrixCRS;
    /// MatrixELL or SplitMatrixELL on the device.
    struct {
      int *length;
      int *index;
      floatType *data;
    } matrixELL;
    /// JacobiCUDA on the device.
    struct {
      floatType *C;
    } jacobi;

    /// Calculate the launch configuration for vectors of length \a N.
    void calculateLaunchConfiguration(int N) {
      blocks = calculateBlocks(N, Threads, MaxBlocks);
      blocksMatvec = calculateBlocks(N, Threads, MaxBlocksMatvec);
    }

    /// @return pointer to the vector on this device.
    virtual floatType *getVector(Vector v) const {
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
  };

  virtual bool supportsMatrixFormat(MatrixFormat format) override {
    return format == MatrixFormatCRS || format == MatrixFormatELL;
  }
  virtual bool supportsPreconditioner(Preconditioner preconditioner) override {
    return preconditioner == PreconditionerJacobi;
  }

  virtual void initJacobi() override {
    jacobi.reset(new JacobiCUDA(*matrixCOO));
  }

  virtual void allocateK() override {
    checkedMallocHost(&k, sizeof(floatType) * N);
  }
  virtual void deallocateK() override { checkedFreeHost(k); }
  virtual void allocateX() override {
    checkedMallocHost(&x, sizeof(floatType) * N);
  }
  virtual void deallocateX() override { checkedFreeHost(x); }

  virtual bool needsTransfer() override { return true; }
  virtual void doTransferTo() override = 0;
  virtual void doTransferFrom() override = 0;

public:
  CGCUDABase() : CG(MatrixFormatELL, PreconditionerJacobi) {}
};

#endif