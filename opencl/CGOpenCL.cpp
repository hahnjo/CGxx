#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>

#include "../CG.h"
#include "../Matrix.h"
#include "../Preconditioner.h"

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include "CL/opencl.h"

const char *source =
#include "kernel.cl"
    ;

/// Class implementing parallel kernels with OpenCL.
class CGOpenCL : public CG {
  cl_device_id device;
  cl_context ctx = NULL;
  cl_command_queue queue = NULL;

  cl_program program = NULL;
  cl_kernel matvecKernelCRS = NULL;
  cl_kernel matvecKernelELL = NULL;
  cl_kernel axpyKernelCL = NULL;
  cl_kernel xpayKernelCL = NULL;
  cl_kernel vectorDotKernelCL = NULL;
  cl_kernel deviceReduceKernel = NULL;
  cl_kernel applyPreconditionerKernelJacobi = NULL;

  const size_t Local = 128;
  const size_t MaxGroups = 1024;
  // 65536 seems to not work on the Pascal nodes.
  const size_t MaxGroupsMatvec = 65535;
  size_t groups;
  size_t global;
  size_t globalMatvec;

  cl_mem tmp = NULL;

  cl_mem k_dev = NULL;
  cl_mem x_dev = NULL;

  cl_mem p_dev = NULL;
  cl_mem q_dev = NULL;
  cl_mem r_dev = NULL;
  cl_mem z_dev = NULL;

  struct {
    cl_mem ptr = NULL;
    cl_mem index = NULL;
    cl_mem value = NULL;
  } matrixCRS_dev;
  struct {
    cl_mem length = NULL;
    cl_mem index = NULL;
    cl_mem data = NULL;
  } matrixELL_dev;
  struct {
    cl_mem C = NULL;
  } jacobi_dev;

  cl_mem getVector(Vector v) {
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

  cl_kernel checkedCreateKernel(const char *kernelName);
  void checkedSetKernelArg(cl_kernel kernel, cl_uint arg_index, size_t arg_size,
                           const void *arg_value);
  void checkedEnqueueNDRangeKernel(cl_kernel kernel, size_t global = 0,
                                   size_t local = 0);

  cl_mem checkedCreateBufferWithFlags(cl_mem_flags flags, size_t size);
  cl_mem checkedCreateBuffer(size_t size);
  cl_mem checkedCreateReadBuffer(size_t size);
  void checkedEnqueWriteBuffer(cl_mem buffer, size_t cb, const void *ptr);
  void checkedReleaseMemObject(cl_mem buffer);

  void checkedFinish();

  void initDevice();
  int getGroups(int maxGroups);
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
  CGOpenCL() : CG(MatrixFormatELL, PreconditionerJacobi) {}
  ~CGOpenCL();
};

static inline void checkError(cl_int error) {
  if (error != CL_SUCCESS) {
    std::cerr << "OpenCL error: " << error << std::endl;
    std::exit(1);
  }
}

cl_kernel CGOpenCL::checkedCreateKernel(const char *kernelName) {
  cl_int err;
  cl_kernel kernel = clCreateKernel(program, kernelName, &err);
  checkError(err);

  return kernel;
}

void CGOpenCL::checkedSetKernelArg(cl_kernel kernel, cl_uint arg_index,
                                   size_t arg_size, const void *arg_value) {
  checkError(clSetKernelArg(kernel, arg_index, arg_size, arg_value));
}

void CGOpenCL::checkedEnqueueNDRangeKernel(cl_kernel kernel, size_t global,
                                           size_t local) {
  if (global == 0) {
    global = this->global;
  }
  if (local == 0) {
    local = Local;
  }

  checkError(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local, 0,
                                    NULL, NULL));
}

cl_mem CGOpenCL::checkedCreateBufferWithFlags(cl_mem_flags flags, size_t size) {
  cl_int err;
  cl_mem mem = clCreateBuffer(ctx, flags, size, NULL, &err);
  checkError(err);

  return mem;
}

cl_mem CGOpenCL::checkedCreateBuffer(size_t size) {
  return checkedCreateBufferWithFlags(CL_MEM_READ_WRITE, size);
}

cl_mem CGOpenCL::checkedCreateReadBuffer(size_t size) {
  return checkedCreateBufferWithFlags(CL_MEM_READ_ONLY, size);
}

void CGOpenCL::checkedEnqueWriteBuffer(cl_mem buffer, size_t cb,
                                       const void *ptr) {
  checkError(
      clEnqueueWriteBuffer(queue, buffer, CL_FALSE, 0, cb, ptr, 0, NULL, NULL));
}

void CGOpenCL::checkedReleaseMemObject(cl_mem buffer) {
  checkError(clReleaseMemObject(buffer));
}

void CGOpenCL::checkedFinish() { checkError(clFinish(queue)); }

void CGOpenCL::initDevice() {
  cl_platform_id platform;
  checkError(clGetPlatformIDs(1, &platform, NULL));

  cl_uint num_devices;
  checkError(
      clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices));

  std::unique_ptr<cl_device_id[]> devices(new cl_device_id[num_devices]);
  checkError(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices,
                            devices.get(), NULL));

  for (cl_uint i = 0; i < num_devices; i++) {
    cl_device_type type;
    checkError(clGetDeviceInfo(devices[i], CL_DEVICE_TYPE,
                               sizeof(cl_device_type), &type, NULL));

    if ((type & CL_DEVICE_TYPE_CPU) == CL_DEVICE_TYPE_CPU) {
      std::cout << "Skipping CPU (device " << i << ")" << std::endl;
      continue;
    }

    device = devices[i];
    return;
  }

  std::cerr << "Could not find OpenCL device!" << std::endl;
  std::exit(1);
}

int CGOpenCL::getGroups(int maxGroups) {
  int maxNeededGroups = (N + Local - 1) / Local;
  int groups = maxNeededGroups, div = 2;

  // We have grid-stride loops so it should be better if all groups receive
  // roughly the same amount of work.
  while (groups > maxGroups) {
    groups = maxNeededGroups / div;
    div++;
  }

  return groups;
}

void CGOpenCL::init(const char *matrixFile) {
  // First init the device and don't read matrix when there is none available.
  initDevice();

  cl_int err;
  ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
  checkError(err);
  queue = clCreateCommandQueue(ctx, device, 0, &err);
  checkError(err);

  // Compile program.
  program = clCreateProgramWithSource(ctx, 1, &source, NULL, &err);
  checkError(err);

  err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
  if (err != CL_SUCCESS) {
    std::cerr << "Error " << err << " for clBuildProgram!" << std::endl;

    size_t size;
    checkError(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0,
                                     NULL, &size));

    std::unique_ptr<char[]> log(new char[size]);
    err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, size,
                                log.get(), NULL);
    if (err == CL_SUCCESS) {
      std::cerr << log.get() << std::endl;
    } else {
      std::cerr << "Error " << err << " for clGetProgramBuildInfo!"
                << std::endl;
    }
    std::exit(1);
  }

  matvecKernelCRS = checkedCreateKernel("matvecKernelCRS");
  matvecKernelELL = checkedCreateKernel("matvecKernelELL");
  axpyKernelCL = checkedCreateKernel("axpyKernel");
  xpayKernelCL = checkedCreateKernel("xpayKernel");
  vectorDotKernelCL = checkedCreateKernel("vectorDotKernel");
  deviceReduceKernel = checkedCreateKernel("deviceReduceKernel");
  applyPreconditionerKernelJacobi =
      checkedCreateKernel("applyPreconditionerKernelJacobi");

  // Now that we have a working device, read the matrix.
  CG::init(matrixFile);

  groups = getGroups(MaxGroups);
  global = groups * Local;
  globalMatvec = getGroups(MaxGroupsMatvec) * Local;
}

void CGOpenCL::doTransferTo() {
  // Allocate memory on the device and transfer necessary data.
  size_t vectorSize = sizeof(floatType) * N;
  k_dev = checkedCreateReadBuffer(vectorSize);
  x_dev = checkedCreateBuffer(vectorSize);
  checkedEnqueWriteBuffer(k_dev, vectorSize, k.get());
  checkedEnqueWriteBuffer(x_dev, vectorSize, x.get());

  p_dev = checkedCreateBuffer(vectorSize);
  q_dev = checkedCreateBuffer(vectorSize);
  r_dev = checkedCreateBuffer(vectorSize);

  switch (matrixFormat) {
  case MatrixFormatCRS: {
    size_t ptrSize = sizeof(int) * (N + 1);
    size_t indexSize = sizeof(int) * nz;
    size_t valueSize = sizeof(floatType) * nz;

    matrixCRS_dev.ptr = checkedCreateBuffer(ptrSize);
    matrixCRS_dev.index = checkedCreateBuffer(indexSize);
    matrixCRS_dev.value = checkedCreateBuffer(valueSize);

    checkedEnqueWriteBuffer(matrixCRS_dev.ptr, ptrSize, matrixCRS->ptr.get());
    checkedEnqueWriteBuffer(matrixCRS_dev.index, indexSize,
                            matrixCRS->index.get());
    checkedEnqueWriteBuffer(matrixCRS_dev.value, valueSize,
                            matrixCRS->value.get());
    break;
  }
  case MatrixFormatELL: {
    int elements = matrixELL->elements;
    size_t lengthSize = sizeof(int) * N;
    size_t indexSize = sizeof(int) * elements;
    size_t dataSize = sizeof(floatType) * elements;

    matrixELL_dev.length = checkedCreateBuffer(lengthSize);
    matrixELL_dev.index = checkedCreateBuffer(indexSize);
    matrixELL_dev.data = checkedCreateBuffer(dataSize);

    checkedEnqueWriteBuffer(matrixELL_dev.length, lengthSize,
                            matrixELL->length.get());
    checkedEnqueWriteBuffer(matrixELL_dev.index, indexSize,
                            matrixELL->index.get());
    checkedEnqueWriteBuffer(matrixELL_dev.data, dataSize,
                            matrixELL->data.get());
    break;
  }
  default:
    assert(0 && "Invalid matrix format!");
  }
  if (preconditioner != PreconditionerNone) {
    z_dev = checkedCreateBuffer(vectorSize);

    switch (preconditioner) {
    case PreconditionerJacobi:
      jacobi_dev.C = checkedCreateBuffer(vectorSize);
      checkedEnqueWriteBuffer(jacobi_dev.C, vectorSize, jacobi->C.get());
      break;
    default:
      assert(0 && "Invalid preconditioner!");
    }
  }

  tmp = checkedCreateBuffer(sizeof(floatType) * MaxGroups);

  checkedFinish();
}

void CGOpenCL::doTransferFrom() {
  // Copy back solution and free memory on the device.
  checkError(clEnqueueReadBuffer(queue, x_dev, CL_FALSE, 0,
                                 sizeof(floatType) * N, x.get(), 0, NULL,
                                 NULL));
  checkedFinish();

  checkedReleaseMemObject(k_dev);
  checkedReleaseMemObject(x_dev);

  checkedReleaseMemObject(p_dev);
  checkedReleaseMemObject(q_dev);
  checkedReleaseMemObject(r_dev);

  switch (matrixFormat) {
  case MatrixFormatCRS: {
    checkedReleaseMemObject(matrixCRS_dev.ptr);
    checkedReleaseMemObject(matrixCRS_dev.index);
    checkedReleaseMemObject(matrixCRS_dev.value);
    break;
  }
  case MatrixFormatELL: {
    checkedReleaseMemObject(matrixELL_dev.length);
    checkedReleaseMemObject(matrixELL_dev.index);
    checkedReleaseMemObject(matrixELL_dev.data);
    break;
  }
  default:
    assert(0 && "Invalid matrix format!");
  }
  if (preconditioner != PreconditionerNone) {
    checkedReleaseMemObject(z_dev);

    switch (preconditioner) {
    case PreconditionerJacobi: {
      checkedReleaseMemObject(jacobi_dev.C);
      break;
    }
    default:
      assert(0 && "Invalid preconditioner!");
    }
  }

  checkedReleaseMemObject(tmp);
}

void CGOpenCL::cpy(Vector _dst, Vector _src) {
  cl_mem dst = getVector(_dst);
  cl_mem src = getVector(_src);

  checkError(clEnqueueCopyBuffer(queue, src, dst, 0, 0, sizeof(floatType) * N,
                                 0, NULL, NULL));
}

void CGOpenCL::matvecKernel(Vector _x, Vector _y) {
  cl_mem x = getVector(_x);
  cl_mem y = getVector(_y);

  switch (matrixFormat) {
  case MatrixFormatCRS:
    checkedSetKernelArg(matvecKernelCRS, 0, sizeof(cl_mem), &matrixCRS_dev.ptr);
    checkedSetKernelArg(matvecKernelCRS, 1, sizeof(cl_mem),
                        &matrixCRS_dev.index);
    checkedSetKernelArg(matvecKernelCRS, 2, sizeof(cl_mem),
                        &matrixCRS_dev.value);
    checkedSetKernelArg(matvecKernelCRS, 3, sizeof(cl_mem), &x);
    checkedSetKernelArg(matvecKernelCRS, 4, sizeof(cl_mem), &y);
    checkedSetKernelArg(matvecKernelCRS, 5, sizeof(int), &N);
    checkedEnqueueNDRangeKernel(matvecKernelCRS, globalMatvec);
    break;
  case MatrixFormatELL:
    checkedSetKernelArg(matvecKernelELL, 0, sizeof(cl_mem),
                        &matrixELL_dev.length);
    checkedSetKernelArg(matvecKernelELL, 1, sizeof(cl_mem),
                        &matrixELL_dev.index);
    checkedSetKernelArg(matvecKernelELL, 2, sizeof(cl_mem),
                        &matrixELL_dev.data);
    checkedSetKernelArg(matvecKernelELL, 3, sizeof(cl_mem), &x);
    checkedSetKernelArg(matvecKernelELL, 4, sizeof(cl_mem), &y);
    checkedSetKernelArg(matvecKernelELL, 5, sizeof(int), &N);
    checkedEnqueueNDRangeKernel(matvecKernelELL, globalMatvec);
    break;
  default:
    assert(0 && "Invalid matrix format!");
  }
  checkedFinish();
}

void CGOpenCL::axpyKernel(floatType a, Vector _x, Vector _y) {
  cl_mem x = getVector(_x);
  cl_mem y = getVector(_y);

  checkedSetKernelArg(axpyKernelCL, 0, sizeof(floatType), &a);
  checkedSetKernelArg(axpyKernelCL, 1, sizeof(cl_mem), &x);
  checkedSetKernelArg(axpyKernelCL, 2, sizeof(cl_mem), &y);
  checkedSetKernelArg(axpyKernelCL, 3, sizeof(int), &N);
  checkedEnqueueNDRangeKernel(axpyKernelCL);
  checkedFinish();
}

void CGOpenCL::xpayKernel(Vector _x, floatType a, Vector _y) {
  cl_mem x = getVector(_x);
  cl_mem y = getVector(_y);

  checkedSetKernelArg(xpayKernelCL, 0, sizeof(cl_mem), &x);
  checkedSetKernelArg(xpayKernelCL, 1, sizeof(floatType), &a);
  checkedSetKernelArg(xpayKernelCL, 2, sizeof(cl_mem), &y);
  checkedSetKernelArg(xpayKernelCL, 3, sizeof(int), &N);
  checkedEnqueueNDRangeKernel(xpayKernelCL);
  checkedFinish();
}

floatType CGOpenCL::vectorDotKernel(Vector _a, Vector _b) {
  floatType res = 0;
  cl_mem a = getVector(_a);
  cl_mem b = getVector(_b);

  // inspired by
  // https://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/
  size_t localForVectorDot = Local * sizeof(floatType);
  size_t localForReduce = MaxGroups * sizeof(floatType);

  checkedSetKernelArg(vectorDotKernelCL, 0, sizeof(cl_mem), &a);
  checkedSetKernelArg(vectorDotKernelCL, 1, sizeof(cl_mem), &b);
  checkedSetKernelArg(vectorDotKernelCL, 2, sizeof(cl_mem), &tmp);
  checkedSetKernelArg(vectorDotKernelCL, 3, localForVectorDot, NULL);
  checkedSetKernelArg(vectorDotKernelCL, 4, sizeof(int), &N);
  checkedEnqueueNDRangeKernel(vectorDotKernelCL);

  checkedSetKernelArg(deviceReduceKernel, 0, sizeof(cl_mem), &tmp);
  checkedSetKernelArg(deviceReduceKernel, 1, sizeof(cl_mem), &tmp);
  checkedSetKernelArg(deviceReduceKernel, 2, localForReduce, NULL);
  checkedSetKernelArg(deviceReduceKernel, 3, sizeof(int), &groups);
  checkedEnqueueNDRangeKernel(deviceReduceKernel, MaxGroups, MaxGroups);

  checkError(clEnqueueReadBuffer(queue, tmp, CL_FALSE, 0, sizeof(floatType),
                                 &res, 0, NULL, NULL));
  checkedFinish();

  return res;
}

void CGOpenCL::applyPreconditionerKernel(Vector _x, Vector _y) {
  cl_mem x = getVector(_x);
  cl_mem y = getVector(_y);

  switch (preconditioner) {
  case PreconditionerJacobi:
    checkedSetKernelArg(applyPreconditionerKernelJacobi, 0, sizeof(cl_mem),
                        &jacobi_dev.C);
    checkedSetKernelArg(applyPreconditionerKernelJacobi, 1, sizeof(cl_mem), &x);
    checkedSetKernelArg(applyPreconditionerKernelJacobi, 2, sizeof(cl_mem), &y);
    checkedSetKernelArg(applyPreconditionerKernelJacobi, 3, sizeof(int), &N);
    checkedEnqueueNDRangeKernel(applyPreconditionerKernelJacobi);
    break;
  default:
    assert(0 && "Invalid preconditioner!");
  }
  checkedFinish();
}

CGOpenCL::~CGOpenCL() {
  clReleaseKernel(matvecKernelCRS);
  clReleaseKernel(matvecKernelELL);
  clReleaseKernel(axpyKernelCL);
  clReleaseKernel(xpayKernelCL);
  clReleaseKernel(vectorDotKernelCL);
  clReleaseKernel(deviceReduceKernel);
  clReleaseKernel(applyPreconditionerKernelJacobi);

  clReleaseProgram(program);
  clReleaseCommandQueue(queue);
  clReleaseContext(ctx);
}

CG *CG::getInstance() { return new CGOpenCL; }
