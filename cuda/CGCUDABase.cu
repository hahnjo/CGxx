#include "CGCUDABase.h"

void CGCUDABase::allocateAndCopyMatrixDataCRS(
    int length, const MatrixDataCRS &data,
    Device::MatrixCRSDevice &deviceMatrix) {
  size_t ptrSize = sizeof(int) * (length + 1);
  int deviceNz = data.ptr[length];
  size_t indexSize = sizeof(int) * deviceNz;
  size_t valueSize = sizeof(floatType) * deviceNz;

  checkedMalloc(&deviceMatrix.ptr, ptrSize);
  checkedMalloc(&deviceMatrix.index, indexSize);
  checkedMalloc(&deviceMatrix.value, valueSize);

  checkedMemcpyAsyncToDevice(deviceMatrix.ptr, data.ptr, ptrSize);
  checkedMemcpyAsyncToDevice(deviceMatrix.index, data.index, indexSize);
  checkedMemcpyAsyncToDevice(deviceMatrix.value, data.value, valueSize);
}

void CGCUDABase::allocateAndCopyMatrixDataELL(
    int length, const MatrixDataELL &data,
    Device::MatrixELLDevice &deviceMatrix) {
  size_t lengthSize = sizeof(int) * length;
  int elements = data.elements;
  size_t indexSize = sizeof(int) * elements;
  size_t dataSize = sizeof(floatType) * elements;

  checkedMalloc(&deviceMatrix.length, lengthSize);
  checkedMalloc(&deviceMatrix.index, indexSize);
  checkedMalloc(&deviceMatrix.data, dataSize);

  checkedMemcpyAsyncToDevice(deviceMatrix.length, data.length, lengthSize);
  checkedMemcpyAsyncToDevice(deviceMatrix.index, data.index, indexSize);
  checkedMemcpyAsyncToDevice(deviceMatrix.data, data.data, dataSize);
}

void CGCUDABase::freeMatrixCRSDevice(
    const Device::MatrixCRSDevice &deviceMatrix) {
  checkedFree(deviceMatrix.ptr);
  checkedFree(deviceMatrix.index);
  checkedFree(deviceMatrix.value);
}

void CGCUDABase::freeMatrixELLDevice(
    const Device::MatrixELLDevice &deviceMatrix) {
  checkedFree(deviceMatrix.length);
  checkedFree(deviceMatrix.index);
  checkedFree(deviceMatrix.data);
}
