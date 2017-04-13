#include <cassert>
#include <cmath>
#include <memory>

#include "../CG.h"
#include "../Matrix.h"
#include "../Preconditioner.h"
#include "CGOpenCLBase.h"
#include "utils.h"

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include "CL/opencl.h"

/// Class implementing parallel kernels with OpenCL.
class CGOpenCL : public CGOpenCLBase {
  static const int ZERO;

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
const int CGOpenCL::ZERO = 0;

void CGOpenCL::init(const char *matrixFile) {
  // First init the device and don't read matrix when there is none available.
  cl_device_id device_id = getAllDevices()[0];

  cl_int err;
  ctx = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
  checkError(err);

  device.init(device_id, this);

  // Now that we have a working device, compile the program and read the matrix.
  CGOpenCLBase::init(matrixFile);

  device.calculateLaunchConfiguration(N);
}

void CGOpenCL::doTransferTo() {
  // Allocate memory on the device and transfer necessary data.
  size_t vectorSize = sizeof(floatType) * N;
  device.k = checkedCreateReadBuffer(vectorSize);
  device.x = checkedCreateBuffer(vectorSize);
  device.checkedEnqueueWriteBuffer(device.k, vectorSize, k);
  device.checkedEnqueueWriteBuffer(device.x, vectorSize, x);

  device.p = checkedCreateBuffer(vectorSize);
  device.q = checkedCreateBuffer(vectorSize);
  device.r = checkedCreateBuffer(vectorSize);

  switch (matrixFormat) {
  case MatrixFormatCRS:
    allocateAndCopyMatrixDataCRS(N, *matrixCRS, device, device.matrixCRS);
    break;
  case MatrixFormatELL:
    allocateAndCopyMatrixDataELL(N, *matrixELL, device, device.matrixELL);
    break;
  default:
    assert(0 && "Invalid matrix format!");
  }
  if (preconditioner != PreconditionerNone) {
    device.z = checkedCreateBuffer(vectorSize);

    switch (preconditioner) {
    case PreconditionerJacobi:
      device.jacobi.C = checkedCreateBuffer(vectorSize);
      device.checkedEnqueueWriteBuffer(device.jacobi.C, vectorSize, jacobi->C);
      break;
    default:
      assert(0 && "Invalid preconditioner!");
    }
  }

  device.tmp = checkedCreateBuffer(sizeof(floatType) * Device::MaxGroups);

  device.checkedFinish();
}

void CGOpenCL::doTransferFrom() {
  // Copy back solution and free memory on the device.
  device.checkedEnqueueReadBuffer(device.x, sizeof(floatType) * N, x);
  device.checkedFinish();

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

void CGOpenCL::cpy(Vector _dst, Vector _src) {
  cl_mem dst = device.getVector(_dst);
  cl_mem src = device.getVector(_src);

  checkError(clEnqueueCopyBuffer(device.queue, src, dst, 0, 0,
                                 sizeof(floatType) * N, 0, NULL, NULL));
  device.checkedFinish();
}

void CGOpenCL::matvecKernel(Vector _x, Vector _y) {
  cl_mem x = device.getVector(_x);
  cl_mem y = device.getVector(_y);

  switch (matrixFormat) {
  case MatrixFormatCRS:
    device.checkedEnqueueMatvecKernelCRS(matvecKernelCRS, device.matrixCRS, x,
                                         y, ZERO, N);
    break;
  case MatrixFormatELL:
    device.checkedEnqueueMatvecKernelELL(matvecKernelELL, device.matrixELL, x,
                                         y, ZERO, N);
    break;
  default:
    assert(0 && "Invalid matrix format!");
  }
  device.checkedFinish();
}

void CGOpenCL::axpyKernel(floatType a, Vector _x, Vector _y) {
  cl_mem x = device.getVector(_x);
  cl_mem y = device.getVector(_y);

  checkedSetKernelArg(axpyKernelCL, 0, sizeof(floatType), &a);
  checkedSetKernelArg(axpyKernelCL, 1, sizeof(cl_mem), &x);
  checkedSetKernelArg(axpyKernelCL, 2, sizeof(int), &ZERO);
  checkedSetKernelArg(axpyKernelCL, 3, sizeof(cl_mem), &y);
  checkedSetKernelArg(axpyKernelCL, 4, sizeof(int), &ZERO);
  checkedSetKernelArg(axpyKernelCL, 5, sizeof(int), &N);
  device.checkedEnqueueNDRangeKernel(axpyKernelCL);
  device.checkedFinish();
}

void CGOpenCL::xpayKernel(Vector _x, floatType a, Vector _y) {
  cl_mem x = device.getVector(_x);
  cl_mem y = device.getVector(_y);

  checkedSetKernelArg(xpayKernelCL, 0, sizeof(cl_mem), &x);
  checkedSetKernelArg(xpayKernelCL, 1, sizeof(int), &ZERO);
  checkedSetKernelArg(xpayKernelCL, 2, sizeof(floatType), &a);
  checkedSetKernelArg(xpayKernelCL, 3, sizeof(cl_mem), &y);
  checkedSetKernelArg(xpayKernelCL, 4, sizeof(int), &ZERO);
  checkedSetKernelArg(xpayKernelCL, 5, sizeof(int), &N);
  device.checkedEnqueueNDRangeKernel(xpayKernelCL);
  device.checkedFinish();
}

floatType CGOpenCL::vectorDotKernel(Vector _a, Vector _b) {
  floatType res = 0;
  cl_mem a = device.getVector(_a);
  cl_mem b = device.getVector(_b);

  // inspired by
  // https://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/
  size_t localForVectorDot = Device::Local * sizeof(floatType);
  size_t localForReduce = Device::MaxGroups * sizeof(floatType);

  checkedSetKernelArg(vectorDotKernelCL, 0, sizeof(cl_mem), &a);
  checkedSetKernelArg(vectorDotKernelCL, 1, sizeof(int), &ZERO);
  checkedSetKernelArg(vectorDotKernelCL, 2, sizeof(cl_mem), &b);
  checkedSetKernelArg(vectorDotKernelCL, 3, sizeof(int), &ZERO);
  checkedSetKernelArg(vectorDotKernelCL, 4, sizeof(cl_mem), &device.tmp);
  checkedSetKernelArg(vectorDotKernelCL, 5, localForVectorDot, NULL);
  checkedSetKernelArg(vectorDotKernelCL, 6, sizeof(int), &N);
  device.checkedEnqueueNDRangeKernel(vectorDotKernelCL);

  checkedSetKernelArg(deviceReduceKernel, 0, sizeof(cl_mem), &device.tmp);
  checkedSetKernelArg(deviceReduceKernel, 1, sizeof(cl_mem), &device.tmp);
  checkedSetKernelArg(deviceReduceKernel, 2, localForReduce, NULL);
  checkedSetKernelArg(deviceReduceKernel, 3, sizeof(int), &device.groups);
  device.checkedEnqueueNDRangeKernel(deviceReduceKernel, Device::MaxGroups,
                                     Device::MaxGroups);

  device.checkedEnqueueReadBuffer(device.tmp, sizeof(floatType), &res);
  device.checkedFinish();

  return res;
}

void CGOpenCL::applyPreconditionerKernel(Vector _x, Vector _y) {
  cl_mem x = device.getVector(_x);
  cl_mem y = device.getVector(_y);

  switch (preconditioner) {
  case PreconditionerJacobi:
    checkedSetKernelArg(applyPreconditionerKernelJacobi, 0, sizeof(cl_mem),
                        &device.jacobi.C);
    checkedSetKernelArg(applyPreconditionerKernelJacobi, 1, sizeof(cl_mem), &x);
    checkedSetKernelArg(applyPreconditionerKernelJacobi, 2, sizeof(int), &ZERO);
    checkedSetKernelArg(applyPreconditionerKernelJacobi, 3, sizeof(cl_mem), &y);
    checkedSetKernelArg(applyPreconditionerKernelJacobi, 4, sizeof(int), &ZERO);
    checkedSetKernelArg(applyPreconditionerKernelJacobi, 5, sizeof(int), &N);
    device.checkedEnqueueNDRangeKernel(applyPreconditionerKernelJacobi);
    break;
  default:
    assert(0 && "Invalid preconditioner!");
  }
  device.checkedFinish();
}

CG *CG::getInstance() { return new CGOpenCL; }
