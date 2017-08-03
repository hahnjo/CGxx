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
    floatType *tmp = nullptr;

    /// CG#VectorK
    floatType *k = nullptr;
    /// CG#VectorX
    floatType *x = nullptr;

    /// CG#VectorP
    floatType *p = nullptr;
    /// CG#VectorQ
    floatType *q = nullptr;
    /// CG#VectorR
    floatType *r = nullptr;
    /// CG#VectorZ
    floatType *z = nullptr;

    /// Struct holding pointers to a MatrixDataCRS on the device.
    struct MatrixCRSDevice {
      /// @see MatrixDataELL#ptr
      int *ptr = nullptr;
      /// @see MatrixDataELL#index
      int *index = nullptr;
      /// @see MatrixDataELL#value
      floatType *value = nullptr;
    };
    /// MatrixDataCRS on the device.
    MatrixCRSDevice matrixCRS;
    /// Struct holding pointers to a MatrixDataELL on the device.
    struct MatrixELLDevice {
      /// @see MatrixDataELL#length
      int *length = nullptr;
      /// @see MatrixDataELL#index
      int *index = nullptr;
      /// @see MatrixDataELL#data
      floatType *data = nullptr;
    };
    /// MatrixDataELL on the device.
    MatrixELLDevice matrixELL;
    /// JacobiCUDA on the device.
    struct {
      floatType *C = nullptr;
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

  /// Allocate and copy \a data on the device.
  void allocateAndCopyMatrixDataCRS(int length, const MatrixDataCRS &data,
                                    Device::MatrixCRSDevice &deviceMatrix);
  /// Allocate and copy \a data on the device.
  void allocateAndCopyMatrixDataELL(int length, const MatrixDataELL &data,
                                    Device::MatrixELLDevice &deviceMatrix);

  /// Free \a deviceMatrix.
  void freeMatrixCRSDevice(const Device::MatrixCRSDevice &deviceMatrix);
  /// Free \a deviceMatrix.
  void freeMatrixELLDevice(const Device::MatrixELLDevice &deviceMatrix);

  virtual bool needsTransfer() override { return true; }
  virtual void doTransferTo() override = 0;
  virtual void doTransferFrom() override = 0;

public:
  /// @see CG
  CGCUDABase(bool overlappedGather = false)
      : CG(MatrixFormatELL, PreconditionerJacobi, overlappedGather) {}
};

#endif
