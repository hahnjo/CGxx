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

#ifndef CG_OPENCL_BASE_H
#define CG_OPENCL_BASE_H

#include <vector>

#include "../CG.h"
#include "utils.h"

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include "CL/opencl.h"

// To be defined in CGOpenCLBase.cpp so that there is exactly one variable!
extern const char *source;

/// Class implementing parallel kernels with OpenCL.
class CGOpenCLBase : public CG {
protected:
  /// The OpenCL context.
  cl_context ctx = NULL;

  /// The OpenCL program containing the kernels.
  cl_program program = NULL;
  /// Kernel for CG#matvec using a MatrixCRS.
  cl_kernel matvecKernelCRS = NULL;
  /// Kernel for CG#matvec using a MatrixELL.
  cl_kernel matvecKernelELL = NULL;
  /// Kernel for CG#axpy.
  cl_kernel axpyKernelCL = NULL;
  /// Kernelf or CG#xpay.
  cl_kernel xpayKernelCL = NULL;
  /// Kernel for CG#vectorDot.
  cl_kernel vectorDotKernelCL = NULL;
  /// Reduction kernel used for CG#vectorDot.
  cl_kernel deviceReduceKernel = NULL;
  /// Kernel for CG#applyPreconditioner using a Jacobi preconditioner.
  cl_kernel applyPreconditionerKernelJacobi = NULL;

  /// Holds information about a single device, especially memory, the command
  /// queue and its launch configuration.
  struct Device {
    /// Local size for all kernels.
    static const size_t Local = 128;
    /// Maximum number of groups for all kernels except CG#matvec.
    static const size_t MaxGroups = 1024;
    /// Maximum number of groups for CG#matvec.
    /// (65536 seems to not work on the Pascal nodes!)
    static const size_t MaxGroupsMatvec = 65535;

    /// Number of threads for all kernels except CG#matvec.
    size_t groups;
    /// Global size for all kernels except CG#matvec.
    size_t global;
    /// Global size for CG#matvec.
    size_t globalMatvec;

    /// This device's id.
    cl_device_id device_id;
    /// The (cached) context.
    cl_context ctx;
    /// The queue for this device.
    cl_command_queue queue = NULL;

    /// Temporary memory for use in reduction of CG#vectorDot.
    cl_mem tmp = NULL;

    /// CG#VectorK
    cl_mem k = NULL;
    /// CG#VectorX
    cl_mem x = NULL;

    /// CG#VectorP
    cl_mem p = NULL;
    /// CG#VectorQ
    cl_mem q = NULL;
    /// CG#VectorR
    cl_mem r = NULL;
    /// CG#VectorZ
    cl_mem z = NULL;

    /// Struct holding pointers to a MatrixDataCRS on the device.
    struct MatrixCRSDevice {
      /// @see MatrixDataELL#ptr
      cl_mem ptr = NULL;
      /// @see MatrixDataELL#index
      cl_mem index = NULL;
      /// @see MatrixDataELL#value
      cl_mem value = NULL;
    };
    /// MatrixDataCRS on the device.
    MatrixCRSDevice matrixCRS;
    /// Struct holding pointers to a MatrixDataELL on the device.
    struct MatrixELLDevice {
      /// @see MatrixDataELL#length
      cl_mem length = NULL;
      /// @see MatrixDataELL#index
      cl_mem index = NULL;
      /// @see MatrixDataELL#data
      cl_mem data = NULL;
    };
    /// MatrixDataELL on the device.
    MatrixELLDevice matrixELL;
    /// JacobiCUDA on the device.
    struct {
      cl_mem C = NULL;
    } jacobi;

    ~Device() { clReleaseCommandQueue(queue); }

    /// Init device with id \a device_id.
    virtual void init(cl_device_id device_id, CGOpenCLBase *cg) {
      this->device_id = device_id;
      this->ctx = cg->ctx;

      cl_int err;
      queue = clCreateCommandQueue(ctx, device_id, 0, &err);
      checkError(err);
    }

    /// Calculate the launch configuration for vectors of length \a N.
    void calculateLaunchConfiguration(int N) {
      groups = calculateGroups(N, Local, MaxGroups);
      global = groups * Local;
      globalMatvec = calculateGroups(N, Local, MaxGroupsMatvec) * Local;
    }

    /// Enqueue \a kernel.
    void checkedEnqueueNDRangeKernel(cl_kernel kernel, size_t global = 0,
                                     size_t local = 0);
    /// Enqueue \a kernel.
    void checkedEnqueueMatvecKernelCRS(cl_kernel kernel,
                                       MatrixCRSDevice &deviceMatrix, cl_mem x,
                                       cl_mem y, int yOffset, int N);
    /// Enqueue \a kernel.
    void checkedEnqueueMatvecKernelELL(cl_kernel kernel,
                                       MatrixELLDevice &deviceMatrix, cl_mem x,
                                       cl_mem y, int yOffset, int N);

    /// Enqueue read of \a buffer.
    void checkedEnqueueReadBuffer(cl_command_queue queue, cl_mem buffer,
                                  size_t offset, size_t cb, void *ptr) {
      checkError(clEnqueueReadBuffer(queue, buffer, CL_FALSE, offset, cb, ptr,
                                     0, NULL, NULL));
    }
    /// Enqueue read of \a buffer.
    void checkedEnqueueReadBuffer(cl_mem buffer, size_t offset, size_t cb,
                                  void *ptr) {
      checkedEnqueueReadBuffer(queue, buffer, offset, cb, ptr);
    }
    /// Enqueue read of \a buffer.
    void checkedEnqueueReadBuffer(cl_mem buffer, size_t cb, void *ptr) {
      checkedEnqueueReadBuffer(buffer, 0, cb, ptr);
    }

    /// Enqueue write of \a buffer.
    void checkedEnqueueWriteBuffer(cl_command_queue queue, cl_mem buffer,
                                   size_t offset, size_t cb, const void *ptr) {
      checkError(clEnqueueWriteBuffer(queue, buffer, CL_FALSE, offset, cb, ptr,
                                      0, NULL, NULL));
    }
    /// Enqueue write of \a buffer.
    void checkedEnqueueWriteBuffer(cl_mem buffer, size_t offset, size_t cb,
                                   const void *ptr) {
      checkedEnqueueWriteBuffer(queue, buffer, offset, cb, ptr);
    }
    /// Enqueue write of \a buffer.
    void checkedEnqueueWriteBuffer(cl_mem buffer, size_t cb, const void *ptr) {
      checkedEnqueueWriteBuffer(buffer, 0, cb, ptr);
    }

    /// Finish all commands in #queue.
    void checkedFinish() { checkError(clFinish(queue)); }

    /// @return memory object of the vector on this device.
    cl_mem getVector(Vector v) {
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
      return NULL;
    }
  };

  virtual bool supportsMatrixFormat(MatrixFormat format) override {
    return format == MatrixFormatCRS || format == MatrixFormatELL;
  }
  virtual bool supportsPreconditioner(Preconditioner preconditioner) override {
    return preconditioner == PreconditionerJacobi;
  }

  /// @return all devices suitable for computation (excluding CPUs).
  static std::vector<cl_device_id> getAllDevices();

  /// @return the loaded kernel called \a kernelname.
  cl_kernel checkedCreateKernel(const char *kernelName);

  /// @return buffer of size \a size created with \a flags.
  cl_mem checkedCreateBufferWithFlags(cl_mem_flags flags, size_t size);
  /// @return read and write buffer.
  /// @see checkedCreateBufferWithFlags
  cl_mem checkedCreateBuffer(size_t size) {
    return checkedCreateBufferWithFlags(CL_MEM_READ_WRITE, size);
  }
  /// @return read only buffer.
  /// @see checkedCreateBufferWithFlags
  cl_mem checkedCreateReadBuffer(size_t size) {
    return checkedCreateBufferWithFlags(CL_MEM_READ_ONLY, size);
  }

  virtual void init(const char *matrixFile) override;

  virtual bool needsTransfer() override { return true; }

  /// Allocate and copy \a data on the \a device.
  void allocateAndCopyMatrixDataCRS(int length, const MatrixDataCRS &data,
                                    Device &device,
                                    Device::MatrixCRSDevice &deviceMatrix);
  /// Allocate and copy \a data on the \a device.
  void allocateAndCopyMatrixDataELL(int length, const MatrixDataELL &data,
                                    Device &device,
                                    Device::MatrixELLDevice &deviceMatrix);

  /// Free \a deviceMatrix.
  void freeMatrixCRSDevice(const Device::MatrixCRSDevice &deviceMatrix);
  /// Free \a deviceMatrix.
  void freeMatrixELLDevice(const Device::MatrixELLDevice &deviceMatrix);

  virtual void cleanup() override;

public:
  /// @see CG
  CGOpenCLBase(bool overlappedGather = false)
      : CG(MatrixFormatELL, PreconditionerJacobi, overlappedGather) {}
};

#endif
