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

#include <algorithm>
#include <cassert>
#include <iostream>
#include <thread>
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

  struct MultiDevice : Device {
    int id;
    CGMultiCUDA *cg;

    MatrixCRSDevice diagMatrixCRS;
    MatrixELLDevice diagMatrixELL;

    floatType vectorDotResult;
    cudaStream_t gatherStream;
    cudaEvent_t gatherFinished;
    cudaStream_t overlappedMatvecStream;

    ~MultiDevice() {
      checkError(cudaStreamDestroy(gatherStream));
      if (cg->overlappedGather) {
        checkError(cudaEventDestroy(gatherFinished));
        checkError(cudaStreamDestroy(overlappedMatvecStream));
      }
    }

    void init(int id, CGMultiCUDA *cg) {
      this->id = id;
      this->cg = cg;

      setDevice();
      checkError(cudaStreamCreate(&gatherStream));
      if (cg->overlappedGather) {
        checkError(cudaEventCreate(&gatherFinished));
        checkError(cudaStreamCreate(&overlappedMatvecStream));
      }
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
  virtual bool supportsOverlappedGather() override { return true; }

  virtual void parseEnvironment() override;
  virtual void init(const char *matrixFile) override;

  void synchronizeAllDevices();
  void synchronizeAllDevicesGatherStream();
  void recordGatherFinished();

  void doTransferToForDevice(int index);
  virtual void doTransferTo() override;
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

public:
  CGMultiCUDA() : CGCUDABase(/* overlappedGather= */ true) {}
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
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

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

void CGMultiCUDA::synchronizeAllDevicesGatherStream() {
  for (const MultiDevice &device : devices) {
    device.setDevice();
    checkError(cudaStreamSynchronize(device.gatherStream));
  }
}

void CGMultiCUDA::recordGatherFinished() {
  assert(overlappedGather == true);

  for (const MultiDevice &device : devices) {
    device.setDevice();
    checkError(cudaEventRecord(device.gatherFinished, device.gatherStream));
  }
}

void CGMultiCUDA::doTransferToForDevice(int index) {
  size_t fullVectorSize = sizeof(floatType) * N;

  MultiDevice &device = devices[index];
  device.setDevice();

  int d = device.id;
  int offset = workDistribution->offsets[d];
  int length = workDistribution->lengths[d];

  size_t vectorSize = sizeof(floatType) * length;
  checkedMalloc(&device.k, vectorSize);
  checkedMalloc(&device.x, fullVectorSize);
  checkedMemcpyToDevice(device.k, k + offset, vectorSize);
  checkedMemcpyToDevice(device.x, x, fullVectorSize);

  checkedMalloc(&device.p, fullVectorSize);
  checkedMalloc(&device.q, vectorSize);
  checkedMalloc(&device.r, vectorSize);

  switch (matrixFormat) {
  case MatrixFormatCRS:
    if (!overlappedGather) {
      allocateAndCopyMatrixDataCRS(length, splitMatrixCRS->data[d],
                                   device.matrixCRS);
    } else {
      allocateAndCopyMatrixDataCRS(length, partitionedMatrixCRS->diag[d],
                                   device.diagMatrixCRS);
      allocateAndCopyMatrixDataCRS(length, partitionedMatrixCRS->minor[d],
                                   device.matrixCRS);
    }
    break;
  case MatrixFormatELL:
    if (!overlappedGather) {
      allocateAndCopyMatrixDataELL(length, splitMatrixELL->data[d],
                                   device.matrixELL);
    } else {
      allocateAndCopyMatrixDataELL(length, partitionedMatrixELL->diag[d],
                                   device.diagMatrixELL);
      allocateAndCopyMatrixDataELL(length, partitionedMatrixELL->minor[d],
                                   device.matrixELL);
    }
    break;
  default:
    assert(0 && "Invalid matrix format!");
  }
  if (preconditioner != PreconditionerNone) {
    checkedMalloc(&device.z, vectorSize);

    switch (preconditioner) {
    case PreconditionerJacobi:
      checkedMalloc(&device.jacobi.C, vectorSize);
      checkedMemcpyToDevice(device.jacobi.C, jacobi->C + offset, vectorSize);
      break;
    default:
      assert(0 && "Invalid preconditioner!");
    }
  }

  checkedMalloc(&device.tmp, sizeof(floatType) * MaxBlocks);
}

void CGMultiCUDA::doTransferTo() {
  int numDevices = getNumberOfChunks();
  std::unique_ptr<std::thread[]> threads(new std::thread[numDevices]);

  // Allocate memory on all devices and transfer necessary data.
  for (int i = 0; i < numDevices; i++) {
    threads[i] = std::thread(&CGMultiCUDA::doTransferToForDevice, this, i);
  }

  // Synchronize started threads.
  for (int i = 0; i < numDevices; i++) {
    threads[i].join();
  }
}

void CGMultiCUDA::doTransferFrom() {
  // Copy back solution and free memory on the device.
  for (MultiDevice &device : devices) {
    device.setDevice();

    int d = device.id;
    int offset = workDistribution->offsets[d];
    int length = workDistribution->lengths[d];

    checkedMemcpy(x + offset, device.x + offset, sizeof(floatType) * length,
                  cudaMemcpyDeviceToHost);

    checkedFree(device.k);
    checkedFree(device.x);

    checkedFree(device.p);
    checkedFree(device.q);
    checkedFree(device.r);

    switch (matrixFormat) {
    case MatrixFormatCRS:
      if (overlappedGather) {
        freeMatrixCRSDevice(device.diagMatrixCRS);
      }
      freeMatrixCRSDevice(device.matrixCRS);
      break;
    case MatrixFormatELL:
      if (overlappedGather) {
        freeMatrixELLDevice(device.diagMatrixELL);
      }
      freeMatrixELLDevice(device.matrixELL);
      break;
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
  synchronizeAllDevicesGatherStream();

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

      checkedMemcpyAsync(x + offset, xHost + offset, sizeof(floatType) * length,
                         cudaMemcpyHostToDevice, device.gatherStream);
    }
  }
  if (!overlappedGather) {
    synchronizeAllDevicesGatherStream();
  } else {
    recordGatherFinished();
  }
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

  if (!overlappedGather) {
    synchronizeAllDevicesGatherStream();
  } else {
    recordGatherFinished();
  }
}

void CGMultiCUDA::matvecKernel(Vector _x, Vector _y) {
  if (overlappedGather) {
    // Start computation on the diagonal that does not require data exchange
    // between the devices. It is efficient to do so before the gather because
    // the computation is expected to take longer. This effectively even hides
    // the overhead of starting the gather.
    for (MultiDevice &device : devices) {
      device.setDevice();

      int length = workDistribution->lengths[device.id];
      floatType *x = device.getVector(_x, /* forMatvec= */ true);
      floatType *y = device.getVector(_y);

      switch (matrixFormat) {
      case MatrixFormatCRS:
        matvecKernelCRS<<<device.blocksMatvec, Threads, 0,
                          device.overlappedMatvecStream>>>(
            device.diagMatrixCRS.ptr, device.diagMatrixCRS.index,
            device.diagMatrixCRS.value, x, y, length);
        break;
      case MatrixFormatELL:
        matvecKernelELL<<<device.blocksMatvec, Threads, 0,
                          device.overlappedMatvecStream>>>(
            device.diagMatrixELL.length, device.diagMatrixELL.index,
            device.diagMatrixELL.data, x, y, length);
        break;
      default:
        assert(0 && "Invalid matrix format!");
      }
      checkLastError();
    }
  }

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
      if (!overlappedGather) {
        matvecKernelCRS<<<device.blocksMatvec, Threads>>>(
            device.matrixCRS.ptr, device.matrixCRS.index,
            device.matrixCRS.value, x, y, length);
      } else {
        // Wait for gather to finish...
        cudaStreamWaitEvent(device.overlappedMatvecStream,
                            device.gatherFinished, 0);

        matvecKernelCRSRoundup<<<device.blocksMatvec, Threads, 0,
                                 device.overlappedMatvecStream>>>(
            device.matrixCRS.ptr, device.matrixCRS.index,
            device.matrixCRS.value, x, y, length);
      }
      break;
    case MatrixFormatELL:
      if (!overlappedGather) {
        matvecKernelELL<<<device.blocksMatvec, Threads>>>(
            device.matrixELL.length, device.matrixELL.index,
            device.matrixELL.data, x, y, length);
      } else {
        // Wait for gather to finish...
        cudaStreamWaitEvent(device.overlappedMatvecStream,
                            device.gatherFinished, 0);

        matvecKernelELLRoundup<<<device.blocksMatvec, Threads, 0,
                                 device.overlappedMatvecStream>>>(
            device.matrixELL.length, device.matrixELL.index,
            device.matrixELL.data, x, y, length);
      }
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

    axpyKernelCUDA<<<device.blocks, Threads>>>(a, x, y, length);
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

    xpayKernelCUDA<<<device.blocks, Threads>>>(x, a, y, length);
    checkLastError();
  }

  synchronizeAllDevices();
}

floatType CGMultiCUDA::vectorDotKernel(Vector _a, Vector _b) {
  // This is needed for warpReduceSum on __CUDA_ARCH__ < 350
  size_t sharedForVectorDot = max(Threads, BlockReduction) * sizeof(floatType);
  size_t sharedForReduce = max(MaxBlocks, BlockReduction) * sizeof(floatType);

  for (MultiDevice &device : devices) {
    device.setDevice();

    int length = workDistribution->lengths[device.id];
    floatType *a = device.getVector(_a);
    floatType *b = device.getVector(_b);

    // https://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/
    vectorDotKernelCUDA<<<device.blocks, Threads, sharedForVectorDot>>>(
        a, b, device.tmp, length);
    checkLastError();
    deviceReduceKernel<<<1, MaxBlocks, sharedForReduce>>>(
        device.tmp, device.tmp, device.blocks);
    checkLastError();
  }

  // Synchronize devices and reduce partial results.
  floatType res = 0;
  for (MultiDevice &device : devices) {
    device.setDevice();
    checkedSynchronize();
    // We cannot queue this asynchronously in the previous loop because
    // device.vectorDotResult is pageable memory!
    checkedMemcpy(&device.vectorDotResult, device.tmp, sizeof(floatType),
                  cudaMemcpyDeviceToHost);
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
      applyPreconditionerKernelJacobi<<<device.blocks, Threads>>>(
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
