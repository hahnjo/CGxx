#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

#include "../CG.h"
#include "CGCUDABase.h"
#include "kernel.h"
#include "utils.h"

/// Class implementing parallel kernels with CUDA.
class CGMultiCUDA : public CGCUDABase {
  enum GatherImpl {
    GatherImplHost,
    GatherImplDevice,
    GatherImplP2P,
  };

  struct SplitMatrixCRSCUDA : SplitMatrixCRS {
    SplitMatrixCRSCUDA(const MatrixCOO &coo, const WorkDistribution &wd)
        : SplitMatrixCRS(coo, wd) {}

    virtual void allocateData(int numberOfChunks) override {
      data.reset((MatrixDataCRS *)new MatrixDataCRSCUDA[numberOfChunks]);
    }
  };
  struct SplitMatrixELLCUDA : SplitMatrixELL {
    SplitMatrixELLCUDA(const MatrixCOO &coo, const WorkDistribution &wd)
        : SplitMatrixELL(coo, wd) {}

    virtual void allocateData(int numberOfChunks) override {
      data.reset((MatrixDataELL *)new MatrixDataELLCUDA[numberOfChunks]);
    }
  };

  struct MultiDevice : Device {
    int id;
    CGMultiCUDA *cg;

    floatType vectorDotResult;
    cudaStream_t gatherStream;

    ~MultiDevice() { checkError(cudaStreamDestroy(gatherStream)); }

    void init(int id, CGMultiCUDA *cg) {
      this->id = id;
      this->cg = cg;

      setDevice();
      checkError(cudaStreamCreate(&gatherStream));
    }
    void setDevice() const { checkedSetDevice(id); }

    floatType *getVector(Vector v) const override {
      return getVector(v, false);
    }
    floatType *getVector(Vector v, bool forMatvec) const {
      assert(!forMatvec || (v == VectorX || v == VectorP));

      floatType *res = Device::getVector(v);
      if (!forMatvec && (v == VectorX || v == VectorP)) {
        // These vectors are fully allocated, but we only need the "local" part.
        res += cg->workDistribution->offsets[id];
      }
      return res;
    }
  };

  std::vector<MultiDevice> devices;
  GatherImpl gatherImpl = GatherImplHost;

  floatType *p = nullptr;

  virtual int getNumberOfChunks() override { return devices.size(); }

  virtual void parseEnvironment() override;
  virtual void init(const char *matrixFile) override;

  virtual void convertToSplitMatrixCRS() {
    splitMatrixCRS.reset(new SplitMatrixCRSCUDA(*matrixCOO, *workDistribution));
  }
  virtual void convertToSplitMatrixELL() {
    splitMatrixELL.reset(new SplitMatrixELLCUDA(*matrixCOO, *workDistribution));
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

  void synchronizeAllDevices();

  void allocateAndCopyMatrixDataCRS(int length, const MatrixDataCRS &data,
                                    Device::MatrixCRSDevice &deviceMatrix);
  void allocateAndCopyMatrixDataELL(int length, const MatrixDataELL &data,
                                    Device::MatrixELLDevice &deviceMatrix);
  virtual void doTransferTo() override;
  void freeMatrixDataCRS(const Device::MatrixCRSDevice &device);
  void freeMatrixDataELL(const Device::MatrixELLDevice &device);
  virtual void doTransferFrom() override;

  virtual void cpy(Vector _dst, Vector _src) override;

  void matvecGatherXViaHost(Vector _x);
  void matvecGatherXOnDevices(Vector _x);
  virtual void matvecKernel(Vector _x, Vector _y) override;

  virtual void axpyKernel(floatType a, Vector _x, Vector _y) override;
  virtual void xpayKernel(Vector _x, floatType a, Vector _y) override;
  virtual floatType vectorDotKernel(Vector _a, Vector _b) override;

  virtual void applyPreconditionerKernel(Vector _x, Vector _y) override;

  virtual void printSummary() override;
  virtual void cleanup() override {
    CG::cleanup();

    if (gatherImpl == GatherImplHost) {
      checkedFreeHost(p);
    }
  }
};

const char *CG_CUDA_GATHER_IMPL = "CG_CUDA_GATHER_IMPL";
const char *CG_CUDA_GATHER_IMPL_HOST = "host";
const char *CG_CUDA_GATHER_IMPL_DEVICE = "device";
const char *CG_CUDA_GATHER_IMPL_P2P = "p2p";

void CGMultiCUDA::parseEnvironment() {
  CG::parseEnvironment();

  const char *env = std::getenv(CG_CUDA_GATHER_IMPL);
  if (env != NULL && *env != 0) {
    std::string lower(env);
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](char c) { return std::tolower(c); });

    if (lower == CG_CUDA_GATHER_IMPL_HOST) {
      gatherImpl = GatherImplHost;
    } else if (lower == CG_CUDA_GATHER_IMPL_DEVICE) {
      gatherImpl = GatherImplDevice;
    } else if (lower == CG_CUDA_GATHER_IMPL_P2P) {
      gatherImpl = GatherImplP2P;
    } else {
      std::cerr << "Invalid value for " << CG_CUDA_GATHER_IMPL << "! ("
                << CG_CUDA_GATHER_IMPL_HOST << ", "
                << CG_CUDA_GATHER_IMPL_DEVICE << ", or "
                << CG_CUDA_GATHER_IMPL_P2P << ")" << std::endl;
      std::exit(1);
    }
  }
}

void CGMultiCUDA::init(const char *matrixFile) {
  int numberOfDevices;
  cudaGetDeviceCount(&numberOfDevices);
  if (numberOfDevices < 2) {
    std::cerr << "Need at least 2 devices!" << std::endl;
    std::exit(1);
  }
  devices.resize(numberOfDevices);
  for (int d = 0; d < numberOfDevices; d++) {
    devices[d].init(d, this);
  }

  // Set each device once for initialization. Enable peer access if requested,
  // or abort if not available.
  // NOTE: cudaDeviceEnablePeerAccess() is unidirectional, so it has to be
  // called once for each direction, twice per combination!
  for (MultiDevice &device : devices) {
    device.setDevice();

    if (gatherImpl == GatherImplP2P) {
      for (MultiDevice &peerDevice : devices) {
        if (peerDevice.id == device.id) {
          continue;
        }

        int canAccessPeer;
        checkError(
            cudaDeviceCanAccessPeer(&canAccessPeer, device.id, peerDevice.id));
        if (!canAccessPeer) {
          std::cerr << "Device " << device.id << " cannot access "
                    << peerDevice.id << "!" << std::endl;
          std::exit(1);
        }

        checkError(cudaDeviceEnablePeerAccess(peerDevice.id, 0));
      }
    }
  }

  CG::init(matrixFile);
  assert(workDistribution->numberOfChunks == numberOfDevices);

  for (int i = 0; i < numberOfDevices; i++) {
    int length = workDistribution->lengths[i];
    devices[i].calculateLaunchConfiguration(length);
  }

  if (gatherImpl == GatherImplHost) {
    checkedMallocHost(&p, sizeof(floatType) * N);
  }
}

void CGMultiCUDA::synchronizeAllDevices() {
  for (const MultiDevice &device : devices) {
    device.setDevice();
    checkedSynchronize();
  }
}

void CGMultiCUDA::allocateAndCopyMatrixDataCRS(
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

void CGMultiCUDA::allocateAndCopyMatrixDataELL(
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

void CGMultiCUDA::doTransferTo() {
  size_t fullVectorSize = sizeof(floatType) * N;

  // Allocate memory on all devices and transfer necessary data.
  for (MultiDevice &device : devices) {
    device.setDevice();

    int d = device.id;
    int offset = workDistribution->offsets[d];
    int length = workDistribution->lengths[d];

    size_t vectorSize = sizeof(floatType) * length;
    checkedMalloc(&device.k, vectorSize);
    checkedMalloc(&device.x, fullVectorSize);
    checkedMemcpyAsyncToDevice(device.k, k + offset, vectorSize);
    checkedMemcpyAsyncToDevice(device.x, x, fullVectorSize);

    checkedMalloc(&device.p, fullVectorSize);
    checkedMalloc(&device.q, vectorSize);
    checkedMalloc(&device.r, vectorSize);

    switch (matrixFormat) {
    case MatrixFormatCRS: {
      allocateAndCopyMatrixDataCRS(length, splitMatrixCRS->data[d],
                                   device.matrixCRS);
      break;
    }
    case MatrixFormatELL: {
      allocateAndCopyMatrixDataELL(length, splitMatrixELL->data[d],
                                   device.matrixELL);
      break;
    }
    default:
      assert(0 && "Invalid matrix format!");
    }
    if (preconditioner != PreconditionerNone) {
      checkedMalloc(&device.z, vectorSize);

      switch (preconditioner) {
      case PreconditionerJacobi:
        checkedMalloc(&device.jacobi.C, vectorSize);
        checkedMemcpyAsyncToDevice(device.jacobi.C, jacobi->C + offset,
                                   vectorSize);
        break;
      default:
        assert(0 && "Invalid preconditioner!");
      }
    }

    checkedMalloc(&device.tmp, sizeof(floatType) * Device::MaxBlocks);
  }

  synchronizeAllDevices();
}

void CGMultiCUDA::freeMatrixDataCRS(const Device::MatrixCRSDevice &device) {
  checkedFree(device.ptr);
  checkedFree(device.index);
  checkedFree(device.value);
}

void CGMultiCUDA::freeMatrixDataELL(const Device::MatrixELLDevice &device) {
  checkedFree(device.length);
  checkedFree(device.index);
  checkedFree(device.data);
}

void CGMultiCUDA::doTransferFrom() {
  // Copy back solution and free memory on the device.
  for (MultiDevice &device : devices) {
    device.setDevice();

    int d = device.id;
    int offset = workDistribution->offsets[d];
    int length = workDistribution->lengths[d];

    checkedMemcpyAsync(x + offset, device.x + offset,
                       sizeof(floatType) * length, cudaMemcpyDeviceToHost);

    checkedFree(device.k);
    checkedFree(device.x);

    checkedFree(device.p);
    checkedFree(device.q);
    checkedFree(device.r);

    switch (matrixFormat) {
    case MatrixFormatCRS: {
      freeMatrixDataCRS(device.matrixCRS);
      break;
    }
    case MatrixFormatELL: {
      freeMatrixDataELL(device.matrixELL);
      break;
    }
    default:
      assert(0 && "Invalid matrix format!");
    }
    if (preconditioner != PreconditionerNone) {
      checkedFree(device.z);

      switch (preconditioner) {
      case PreconditionerJacobi: {
        checkedFree(device.jacobi.C);
        break;
      }
      default:
        assert(0 && "Invalid preconditioner!");
      }
    }

    checkedFree(device.tmp);
  }

  synchronizeAllDevices();
}

void CGMultiCUDA::cpy(Vector _dst, Vector _src) {
  for (MultiDevice &device : devices) {
    device.setDevice();

    int length = workDistribution->lengths[device.id];
    floatType *dst = device.getVector(_dst);
    floatType *src = device.getVector(_src);

    checkedMemcpyAsync(dst, src, sizeof(floatType) * length,
                       cudaMemcpyDeviceToDevice);
  }

  synchronizeAllDevices();
}

void CGMultiCUDA::matvecGatherXViaHost(Vector _x) {
  floatType *xHost;
  switch (_x) {
  case VectorX:
    xHost = x;
    break;
  case VectorP:
    xHost = p;
    break;
  default:
    assert(0 && "Invalid vector!");
    return;
  }

  // Gather x on host.
  for (MultiDevice &device : devices) {
    device.setDevice();

    int offset = workDistribution->offsets[device.id];
    int length = workDistribution->lengths[device.id];
    floatType *x = device.getVector(_x);

    checkedMemcpyAsync(xHost + offset, x, sizeof(floatType) * length,
                       cudaMemcpyDeviceToHost, device.gatherStream);
  }
  synchronizeAllDevices();

  // Transfer x to devices.
  for (MultiDevice &device : devices) {
    device.setDevice();

    floatType *x = device.getVector(_x, /* forMatvec= */ true);

    for (MultiDevice &src : devices) {
      if (src.id == device.id) {
        // Don't transfer chunk that is already on the device.
        continue;
      }
      int offset = workDistribution->offsets[src.id];
      int length = workDistribution->lengths[src.id];

      checkedMemcpyAsyncToDevice(x + offset, xHost + offset,
                                 sizeof(floatType) * length,
                                 device.gatherStream);
    }
  }
  synchronizeAllDevices();
}

void CGMultiCUDA::matvecGatherXOnDevices(Vector _x) {
  for (MultiDevice &device : devices) {
    device.setDevice();
    floatType *x = device.getVector(_x, /* forMatvec= */ true);

    for (MultiDevice &src : devices) {
      if (src.id == device.id) {
        // Don't transfer chunk that is already on the device.
        continue;
      }

      int offset = workDistribution->offsets[src.id];
      int length = workDistribution->lengths[src.id];
      floatType *xSrc = src.getVector(_x, /* forMatvec= */ true);

      checkedMemcpyAsync(x + offset, xSrc + offset, sizeof(floatType) * length,
                         cudaMemcpyDeviceToDevice, device.gatherStream);
    }
  }

  synchronizeAllDevices();
}

void CGMultiCUDA::matvecKernel(Vector _x, Vector _y) {
  switch (gatherImpl) {
  case GatherImplHost:
    matvecGatherXViaHost(_x);
    break;
  case GatherImplDevice:
  case GatherImplP2P:
    matvecGatherXOnDevices(_x);
    break;
  default:
    assert(0 && "Invalid gather implementation!");
  }

  for (MultiDevice &device : devices) {
    device.setDevice();

    int length = workDistribution->lengths[device.id];
    floatType *x = device.getVector(_x, /* forMatvec= */ true);
    floatType *y = device.getVector(_y);

    switch (matrixFormat) {
    case MatrixFormatCRS:
      matvecKernelCRS<<<device.blocksMatvec, Device::Threads>>>(
          device.matrixCRS.ptr, device.matrixCRS.index, device.matrixCRS.value,
          x, y, length);
      break;
    case MatrixFormatELL:
      matvecKernelELL<<<device.blocksMatvec, Device::Threads>>>(
          device.matrixELL.length, device.matrixELL.index,
          device.matrixELL.data, x, y, length);
      break;
    default:
      assert(0 && "Invalid matrix format!");
    }
    checkLastError();
  }

  synchronizeAllDevices();
}

void CGMultiCUDA::axpyKernel(floatType a, Vector _x, Vector _y) {
  for (MultiDevice &device : devices) {
    device.setDevice();

    int length = workDistribution->lengths[device.id];
    floatType *x = device.getVector(_x);
    floatType *y = device.getVector(_y);

    axpyKernelCUDA<<<device.blocks, Device::Threads>>>(a, x, y, length);
    checkLastError();
  }

  synchronizeAllDevices();
}

void CGMultiCUDA::xpayKernel(Vector _x, floatType a, Vector _y) {
  for (MultiDevice &device : devices) {
    device.setDevice();

    int length = workDistribution->lengths[device.id];
    floatType *x = device.getVector(_x);
    floatType *y = device.getVector(_y);

    xpayKernelCUDA<<<device.blocks, Device::Threads>>>(x, a, y, length);
    checkLastError();
  }

  synchronizeAllDevices();
}

floatType CGMultiCUDA::vectorDotKernel(Vector _a, Vector _b) {
  // This is needed for warpReduceSum on __CUDA_ARCH__ < 350
  size_t sharedForVectorDot =
      max(Device::Threads, BlockReduction) * sizeof(floatType);
  size_t sharedForReduce =
      max(Device::MaxBlocks, BlockReduction) * sizeof(floatType);

  for (MultiDevice &device : devices) {
    device.setDevice();

    int length = workDistribution->lengths[device.id];
    floatType *a = device.getVector(_a);
    floatType *b = device.getVector(_b);

    // https://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/
    vectorDotKernelCUDA<<<device.blocks, Device::Threads, sharedForVectorDot>>>(
        a, b, device.tmp, length);
    checkLastError();
    deviceReduceKernel<<<1, Device::MaxBlocks, sharedForReduce>>>(
        device.tmp, device.tmp, device.blocks);
    checkLastError();

    checkedMemcpyAsync(&device.vectorDotResult, device.tmp, sizeof(floatType),
                       cudaMemcpyDeviceToHost);
  }

  // Synchronize devices and reduce partial results.
  floatType res = 0;
  for (MultiDevice &device : devices) {
    device.setDevice();
    checkedSynchronize();
    res += device.vectorDotResult;
  }

  return res;
}

void CGMultiCUDA::applyPreconditionerKernel(Vector _x, Vector _y) {
  for (MultiDevice &device : devices) {
    device.setDevice();

    int length = workDistribution->lengths[device.id];
    floatType *x = device.getVector(_x);
    floatType *y = device.getVector(_y);

    switch (preconditioner) {
    case PreconditionerJacobi:
      applyPreconditionerKernelJacobi<<<device.blocks, Device::Threads>>>(
          device.jacobi.C, x, y, length);
      break;
    default:
      assert(0 && "Invalid preconditioner!");
    }
    checkLastError();
  }

  synchronizeAllDevices();
}

void CGMultiCUDA::printSummary() {
  CG::printSummary();

  std::cout << std::endl;
  std::string gatherImplName;
  switch (gatherImpl) {
  case GatherImplHost:
    gatherImplName = "via host";
    break;
  case GatherImplDevice:
    gatherImplName = "between devices, but no peer-to-peer";
    break;
  case GatherImplP2P:
    gatherImplName = "peer-to-peer (NVLink)";
    break;
  }
  assert(gatherImplName.length() > 0);
  printPadded("Gather implementation:", gatherImplName);
}

CG *CG::getInstance() { return new CGMultiCUDA; }
