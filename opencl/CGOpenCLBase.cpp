#include <iostream>
#include <memory>
#include <vector>

#include "CGOpenCLBase.h"

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include "CL/opencl.h"

const char *source =
#include "kernel.cl"
    ;

std::vector<cl_device_id> CGOpenCLBase::getAllDevices() {
  cl_platform_id platform;
  checkError(clGetPlatformIDs(1, &platform, NULL));

  cl_uint num_devices;
  checkError(
      clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices));

  std::unique_ptr<cl_device_id[]> queriedDevices(new cl_device_id[num_devices]);
  checkError(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices,
                            queriedDevices.get(), NULL));

  std::vector<cl_device_id> devices;
  for (cl_uint i = 0; i < num_devices; i++) {
    cl_device_type type;
    checkError(clGetDeviceInfo(queriedDevices[i], CL_DEVICE_TYPE,
                               sizeof(cl_device_type), &type, NULL));

    if ((type & CL_DEVICE_TYPE_CPU) == CL_DEVICE_TYPE_CPU) {
      std::cout << "Skipping CPU (device " << i << ")" << std::endl;
      continue;
    }

    devices.push_back(queriedDevices[i]);
  }

  if (devices.size() == 0) {
    std::cerr << "Could not find any suitable OpenCL device!" << std::endl;
    std::exit(1);
  }

  return devices;
}

cl_kernel CGOpenCLBase::checkedCreateKernel(const char *kernelName) {
  cl_int err;
  cl_kernel kernel = clCreateKernel(program, kernelName, &err);
  checkError(err);

  return kernel;
}

cl_mem CGOpenCLBase::checkedCreateBufferWithFlags(cl_mem_flags flags,
                                                  size_t size) {
  cl_int err;
  cl_mem mem = clCreateBuffer(ctx, flags, size, NULL, &err);
  checkError(err);

  return mem;
}

void CGOpenCLBase::init(const char *matrixFile) {
  // Compile program.
  cl_int err;
  program = clCreateProgramWithSource(ctx, 1, &source, NULL, &err);
  checkError(err);

  err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  if (err != CL_SUCCESS) {
    std::cerr << "Error " << err << " for clBuildProgram!" << std::endl;

    cl_device_id device;
    checkError(clGetProgramInfo(program, CL_PROGRAM_DEVICES,
                                sizeof(cl_device_id), &device, NULL));

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
}

void CGOpenCLBase::allocateAndCopyMatrixDataCRS(
    int length, const MatrixDataCRS &data, Device &device,
    Device::MatrixCRSDevice &deviceMatrix) {
  size_t ptrSize = sizeof(int) * (length + 1);
  int deviceNz = data.ptr[length];
  size_t indexSize = sizeof(int) * deviceNz;
  size_t valueSize = sizeof(floatType) * deviceNz;

  deviceMatrix.ptr = checkedCreateBuffer(ptrSize);
  deviceMatrix.index = checkedCreateBuffer(indexSize);
  deviceMatrix.value = checkedCreateBuffer(valueSize);

  device.checkedEnqueueWriteBuffer(deviceMatrix.ptr, ptrSize, data.ptr);
  device.checkedEnqueueWriteBuffer(deviceMatrix.index, indexSize, data.index);
  device.checkedEnqueueWriteBuffer(deviceMatrix.value, valueSize, data.value);
}

void CGOpenCLBase::allocateAndCopyMatrixDataELL(
    int length, const MatrixDataELL &data, Device &device,
    Device::MatrixELLDevice &deviceMatrix) {
  size_t lengthSize = sizeof(int) * length;
  int elements = data.elements;
  size_t indexSize = sizeof(int) * elements;
  size_t dataSize = sizeof(floatType) * elements;

  deviceMatrix.length = checkedCreateBuffer(lengthSize);
  deviceMatrix.index = checkedCreateBuffer(indexSize);
  deviceMatrix.data = checkedCreateBuffer(dataSize);

  device.checkedEnqueueWriteBuffer(deviceMatrix.length, lengthSize,
                                   data.length);
  device.checkedEnqueueWriteBuffer(deviceMatrix.index, indexSize, data.index);
  device.checkedEnqueueWriteBuffer(deviceMatrix.data, dataSize, data.data);
}

void CGOpenCLBase::freeMatrixCRSDevice(
    const Device::MatrixCRSDevice &deviceMatrix) {
  checkedReleaseMemObject(deviceMatrix.ptr);
  checkedReleaseMemObject(deviceMatrix.index);
  checkedReleaseMemObject(deviceMatrix.value);
}

void CGOpenCLBase::freeMatrixELLDevice(
    const Device::MatrixELLDevice &deviceMatrix) {
  checkedReleaseMemObject(deviceMatrix.length);
  checkedReleaseMemObject(deviceMatrix.index);
  checkedReleaseMemObject(deviceMatrix.data);
}

CGOpenCLBase::~CGOpenCLBase() {
  clReleaseKernel(matvecKernelCRS);
  clReleaseKernel(matvecKernelELL);
  clReleaseKernel(axpyKernelCL);
  clReleaseKernel(xpayKernelCL);
  clReleaseKernel(vectorDotKernelCL);
  clReleaseKernel(deviceReduceKernel);
  clReleaseKernel(applyPreconditionerKernelJacobi);

  clReleaseProgram(program);
  clReleaseContext(ctx);
}
