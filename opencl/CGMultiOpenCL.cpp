#include <cassert>
#include <cmath>
#include <memory>
#include <vector>

#include "../CG.h"
#include "../Matrix.h"
#include "../Preconditioner.h"
#include "../WorkDistribution.h"
#include "CGOpenCLBase.h"
#include "utils.h"

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include "CL/opencl.h"

/// Class implementing parallel kernels with OpenCL.
class CGMultiOpenCL : public CGOpenCLBase {
  struct MultiDevice : Device {
    int id;
    WorkDistribution *workDistribution;

    floatType vectorDotResult;

    int getOffset(Vector v) const {
      if (v == VectorX || v == VectorP) {
        // These vectors are fully allocated, but we only need the "local" part.
        return workDistribution->offsets[id];
      }

      return 0;
    }
  };

  std::vector<MultiDevice> devices;

  std::unique_ptr<floatType[]> p;

  virtual int getNumberOfChunks() override { return devices.size(); }

  virtual void init(const char *matrixFile) override;

  void finishAllDevices();

  virtual void doTransferTo() override;
  virtual void doTransferFrom() override;

  virtual void cpy(Vector _dst, Vector _src) override;

  void matvecGatherXViaHost(Vector _x);

  virtual void matvecKernel(Vector _x, Vector _y) override;

  virtual void axpyKernel(floatType a, Vector _x, Vector _y) override;
  virtual void xpayKernel(Vector _x, floatType a, Vector _y) override;
  virtual floatType vectorDotKernel(Vector _a, Vector _b) override;

  virtual void applyPreconditionerKernel(Vector _x, Vector _y) override;
};

void CGMultiOpenCL::init(const char *matrixFile) {
  // Init all devices and don't read matrix when there is none available.
  std::vector<cl_device_id> device_ids = getAllDevices();
  int numberOfDevices = device_ids.size();

  cl_int err;
  ctx = clCreateContext(NULL, numberOfDevices, device_ids.data(), NULL, NULL,
                        &err);
  checkError(err);

  devices.resize(numberOfDevices);
  for (int d = 0; d < numberOfDevices; d++) {
    devices[d].id = d;
    devices[d].init(device_ids[d], this);
  }

  // Now that we have working devices, compile the program and read the matrix.
  CGOpenCLBase::init(matrixFile);
  assert(workDistribution->numberOfChunks == numberOfDevices);

  for (MultiDevice &device : devices) {
    device.workDistribution = workDistribution.get();
    int length = workDistribution->lengths[device.id];
    device.calculateLaunchConfiguration(length);
  }

  p.reset(new floatType[N]);
}

void CGMultiOpenCL::finishAllDevices() {
  for (MultiDevice &device : devices) {
    device.checkedFinish();
  }
}

void CGMultiOpenCL::doTransferTo() {
  size_t fullVectorSize = sizeof(floatType) * N;

  // Allocate memory on all devices and transfer necessary data.
  for (MultiDevice &device : devices) {
    int d = device.id;
    int offset = workDistribution->offsets[d];
    int length = workDistribution->lengths[d];

    size_t vectorSize = sizeof(floatType) * length;
    device.k = checkedCreateReadBuffer(vectorSize);
    device.x = checkedCreateBuffer(fullVectorSize);
    device.checkedEnqueueWriteBuffer(device.k, vectorSize, k + offset);
    device.checkedEnqueueWriteBuffer(device.x, fullVectorSize, x);

    device.p = checkedCreateBuffer(fullVectorSize);
    device.q = checkedCreateBuffer(vectorSize);
    device.r = checkedCreateBuffer(vectorSize);

    switch (matrixFormat) {
    case MatrixFormatCRS:
      allocateAndCopyMatrixDataCRS(length, splitMatrixCRS->data[d], device,
                                   device.matrixCRS);
      break;
    case MatrixFormatELL:
      allocateAndCopyMatrixDataELL(length, splitMatrixELL->data[d], device,
                                   device.matrixELL);
      break;
    default:
      assert(0 && "Invalid matrix format!");
    }
    if (preconditioner != PreconditionerNone) {
      device.z = checkedCreateBuffer(vectorSize);

      switch (preconditioner) {
      case PreconditionerJacobi:
        device.jacobi.C = checkedCreateBuffer(vectorSize);
        device.checkedEnqueueWriteBuffer(device.jacobi.C, vectorSize,
                                         jacobi->C + offset);
        break;
      default:
        assert(0 && "Invalid preconditioner!");
      }
    }

    device.tmp = checkedCreateBuffer(sizeof(floatType) * Device::MaxGroups);
  }

  finishAllDevices();
}

void CGMultiOpenCL::doTransferFrom() {
  // Copy back solution and free memory on the devices.
  for (MultiDevice &device : devices) {
    int d = device.id;
    int offset = workDistribution->offsets[d];
    int length = workDistribution->lengths[d];

    device.checkedEnqueueReadBuffer(device.x, sizeof(floatType) * offset,
                                    sizeof(floatType) * length, x + offset);

    checkedReleaseMemObject(device.k);
    checkedReleaseMemObject(device.x);

    checkedReleaseMemObject(device.p);
    checkedReleaseMemObject(device.q);
    checkedReleaseMemObject(device.r);

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
      checkedReleaseMemObject(device.z);

      switch (preconditioner) {
      case PreconditionerJacobi: {
        checkedReleaseMemObject(device.jacobi.C);
        break;
      }
      default:
        assert(0 && "Invalid preconditioner!");
      }
    }

    checkedReleaseMemObject(device.tmp);
  }

  finishAllDevices();
}

void CGMultiOpenCL::cpy(Vector _dst, Vector _src) {
  for (MultiDevice &device : devices) {
    int length = workDistribution->lengths[device.id];
    cl_mem dst = device.getVector(_dst);
    int dstOffset = device.getOffset(_dst);
    cl_mem src = device.getVector(_src);
    int srcOffset = device.getOffset(_src);

    checkError(clEnqueueCopyBuffer(device.queue, src, dst,
                                   sizeof(floatType) * srcOffset,
                                   sizeof(floatType) * dstOffset,
                                   sizeof(floatType) * length, 0, NULL, NULL));
  }

  finishAllDevices();
}

void CGMultiOpenCL::matvecGatherXViaHost(Vector _x) {
  floatType *xHost;
  switch (_x) {
  case VectorX:
    xHost = x;
    break;
  case VectorP:
    xHost = p.get();
    break;
  default:
    assert(0 && "Invalid vector!");
    return;
  }

  // Gather x on host.
  for (MultiDevice &device : devices) {
    int offset = workDistribution->offsets[device.id];
    int length = workDistribution->lengths[device.id];
    cl_mem x = device.getVector(_x);
    assert(offset == device.getOffset(_x));

    device.checkedEnqueueReadBuffer(x, sizeof(floatType) * offset,
                                    sizeof(floatType) * length, xHost + offset);
  }
  finishAllDevices();

  // Transfer x to devices.
  for (MultiDevice &device : devices) {
    cl_mem x = device.getVector(_x);

    for (MultiDevice &src : devices) {
      if (src.id == device.id) {
        // Don't transfer chunk that is already on the device.
        continue;
      }
      int offset = workDistribution->offsets[src.id];
      int length = workDistribution->lengths[src.id];

      device.checkedEnqueueWriteBuffer(x, sizeof(floatType) * offset,
                                       sizeof(floatType) * length,
                                       xHost + offset);
    }
  }
  finishAllDevices();
}

void CGMultiOpenCL::matvecKernel(Vector _x, Vector _y) {
  matvecGatherXViaHost(_x);

  for (MultiDevice &device : devices) {
    int length = workDistribution->lengths[device.id];
    cl_mem x = device.getVector(_x);
    cl_mem y = device.getVector(_y);
    int yOffset = device.getOffset(_y);

    switch (matrixFormat) {
    case MatrixFormatCRS:
      device.checkedEnqueueMatvecKernelCRS(matvecKernelCRS, device.matrixCRS, x,
                                           y, yOffset, length);
      break;
    case MatrixFormatELL:
      device.checkedEnqueueMatvecKernelELL(matvecKernelELL, device.matrixELL, x,
                                           y, yOffset, length);
      break;
    default:
      assert(0 && "Invalid matrix format!");
    }
  }

  finishAllDevices();
}

void CGMultiOpenCL::axpyKernel(floatType a, Vector _x, Vector _y) {
  for (MultiDevice &device : devices) {
    int length = workDistribution->lengths[device.id];
    cl_mem x = device.getVector(_x);
    int xOffset = device.getOffset(_x);
    cl_mem y = device.getVector(_y);
    int yOffset = device.getOffset(_y);

    checkedSetKernelArg(axpyKernelCL, 0, sizeof(floatType), &a);
    checkedSetKernelArg(axpyKernelCL, 1, sizeof(cl_mem), &x);
    checkedSetKernelArg(axpyKernelCL, 2, sizeof(int), &xOffset);
    checkedSetKernelArg(axpyKernelCL, 3, sizeof(cl_mem), &y);
    checkedSetKernelArg(axpyKernelCL, 4, sizeof(int), &yOffset);
    checkedSetKernelArg(axpyKernelCL, 5, sizeof(int), &length);
    device.checkedEnqueueNDRangeKernel(axpyKernelCL);
  }

  finishAllDevices();
}

void CGMultiOpenCL::xpayKernel(Vector _x, floatType a, Vector _y) {
  for (MultiDevice &device : devices) {
    int length = workDistribution->lengths[device.id];
    cl_mem x = device.getVector(_x);
    int xOffset = device.getOffset(_x);
    cl_mem y = device.getVector(_y);
    int yOffset = device.getOffset(_y);

    checkedSetKernelArg(xpayKernelCL, 0, sizeof(cl_mem), &x);
    checkedSetKernelArg(xpayKernelCL, 1, sizeof(int), &xOffset);
    checkedSetKernelArg(xpayKernelCL, 2, sizeof(floatType), &a);
    checkedSetKernelArg(xpayKernelCL, 3, sizeof(cl_mem), &y);
    checkedSetKernelArg(xpayKernelCL, 4, sizeof(int), &yOffset);
    checkedSetKernelArg(xpayKernelCL, 5, sizeof(int), &length);
    device.checkedEnqueueNDRangeKernel(xpayKernelCL);
  }

  finishAllDevices();
}

floatType CGMultiOpenCL::vectorDotKernel(Vector _a, Vector _b) {
  size_t localForVectorDot = Device::Local * sizeof(floatType);
  size_t localForReduce = Device::MaxGroups * sizeof(floatType);

  for (MultiDevice &device : devices) {
    int length = workDistribution->lengths[device.id];
    cl_mem a = device.getVector(_a);
    int aOffset = device.getOffset(_a);
    cl_mem b = device.getVector(_b);
    int bOffset = device.getOffset(_b);

    // inspired by
    // https://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/

    checkedSetKernelArg(vectorDotKernelCL, 0, sizeof(cl_mem), &a);
    checkedSetKernelArg(vectorDotKernelCL, 1, sizeof(int), &aOffset);
    checkedSetKernelArg(vectorDotKernelCL, 2, sizeof(cl_mem), &b);
    checkedSetKernelArg(vectorDotKernelCL, 3, sizeof(int), &bOffset);
    checkedSetKernelArg(vectorDotKernelCL, 4, sizeof(cl_mem), &device.tmp);
    checkedSetKernelArg(vectorDotKernelCL, 5, localForVectorDot, NULL);
    checkedSetKernelArg(vectorDotKernelCL, 6, sizeof(int), &length);
    device.checkedEnqueueNDRangeKernel(vectorDotKernelCL);

    checkedSetKernelArg(deviceReduceKernel, 0, sizeof(cl_mem), &device.tmp);
    checkedSetKernelArg(deviceReduceKernel, 1, sizeof(cl_mem), &device.tmp);
    checkedSetKernelArg(deviceReduceKernel, 2, localForReduce, NULL);
    checkedSetKernelArg(deviceReduceKernel, 3, sizeof(int), &device.groups);
    device.checkedEnqueueNDRangeKernel(deviceReduceKernel, Device::MaxGroups,
                                       Device::MaxGroups);

    device.checkedEnqueueReadBuffer(device.tmp, sizeof(floatType),
                                    &device.vectorDotResult);
  }

  // Synchronize devices and reduce partial results.
  floatType res = 0;
  for (MultiDevice &device : devices) {
    device.checkedFinish();
    res += device.vectorDotResult;
  }

  return res;
}

void CGMultiOpenCL::applyPreconditionerKernel(Vector _x, Vector _y) {
  for (MultiDevice &device : devices) {
    int length = workDistribution->lengths[device.id];
    cl_mem x = device.getVector(_x);
    int xOffset = device.getOffset(_x);
    cl_mem y = device.getVector(_y);
    int yOffset = device.getOffset(_y);

    switch (preconditioner) {
    case PreconditionerJacobi:
      checkedSetKernelArg(applyPreconditionerKernelJacobi, 0, sizeof(cl_mem),
                          &device.jacobi.C);
      checkedSetKernelArg(applyPreconditionerKernelJacobi, 1, sizeof(cl_mem),
                          &x);
      checkedSetKernelArg(applyPreconditionerKernelJacobi, 2, sizeof(int),
                          &xOffset);
      checkedSetKernelArg(applyPreconditionerKernelJacobi, 3, sizeof(cl_mem),
                          &y);
      checkedSetKernelArg(applyPreconditionerKernelJacobi, 4, sizeof(int),
                          &yOffset);
      checkedSetKernelArg(applyPreconditionerKernelJacobi, 5, sizeof(int),
                          &length);
      device.checkedEnqueueNDRangeKernel(applyPreconditionerKernelJacobi);
      break;
    default:
      assert(0 && "Invalid preconditioner!");
    }
  }

  finishAllDevices();
}

CG *CG::getInstance() { return new CGMultiOpenCL; }
