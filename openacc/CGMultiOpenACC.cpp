#include <cassert>
#include <memory>

#include <openacc.h>

#include "../CG.h"
#include "../Matrix.h"
#include "../Preconditioner.h"

static inline int getNumberOfDevices() {
  return acc_get_num_devices(acc_get_device_type());
}

/// Class implementing parallel kernels with OpenACC.
class CGMultiOpenACC : public CG {
  std::unique_ptr<floatType[]> p;
  std::unique_ptr<floatType[]> q;
  std::unique_ptr<floatType[]> r;
  std::unique_ptr<floatType[]> z;

  floatType *getVector(Vector v) {
    switch (v) {
    case VectorK:
      return k;
    case VectorX:
      return x;
    case VectorP:
      return p.get();
    case VectorQ:
      return q.get();
    case VectorR:
      return r.get();
    case VectorZ:
      return z.get();
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

  virtual int getNumberOfChunks() override { return getNumberOfDevices(); }

  virtual void init(const char *matrixFile) override;

  virtual bool needsTransfer() override { return true; }
  virtual void doTransferTo() override;
  virtual void doTransferFrom() override;

  virtual void cpy(Vector _dst, Vector _src) override;

  void matvecGatherXViaHost(floatType *x);
  void matvecKernelCRS(floatType *x, floatType *y);
  void matvecKernelELL(floatType *x, floatType *y);

  virtual void matvecKernel(Vector _x, Vector _y) override;
  virtual void axpyKernel(floatType a, Vector _x, Vector _y) override;
  virtual void xpayKernel(Vector _x, floatType a, Vector _y) override;
  virtual floatType vectorDotKernel(Vector _a, Vector _b) override;

  void applyPreconditionerKernelJacobi(floatType *x, floatType *y);

  virtual void applyPreconditionerKernel(Vector _x, Vector _y) override;

public:
  CGMultiOpenACC() : CG(MatrixFormatELL, PreconditionerJacobi) {}
};

void CGMultiOpenACC::init(const char *matrixFile) {
  // First init the device(s).
  acc_init(acc_get_device_type());

  CG::init(matrixFile);
  assert(workDistribution->numberOfChunks == getNumberOfDevices());

  p.reset(new floatType[N]);
  q.reset(new floatType[N]);
  r.reset(new floatType[N]);
  if (preconditioner != PreconditionerNone) {
    z.reset(new floatType[N]);
  }
}

static inline void waitForAllDevices() {
  for (int d = 0; d < getNumberOfDevices(); d++) {
    acc_set_device_num(d, acc_get_device_type());
    #pragma acc wait
  }
}

static inline void enterMatrixCRS(const MatrixDataCRS &matrix, int N) {
  int *ptr = matrix.ptr;
  int nz = ptr[N];
  int *index = matrix.index;
  floatType *value = matrix.value;

  #pragma acc enter data copyin(ptr[0:N+1], index[0:nz], value[0:nz])
}

static inline void enterMatrixELL(const MatrixDataELL &matrix, int N) {
  int elements = matrix.elements;
  int *length = matrix.length;
  int *index = matrix.index;
  floatType *data = matrix.data;

  #pragma acc enter data async copyin(length[0:N], index[0:elements]) \
                               copyin(data[0:elements])
}

void CGMultiOpenACC::doTransferTo() {
  for (int d = 0; d < getNumberOfDevices(); d++) {
    acc_set_device_num(d, acc_get_device_type());
    int offset = workDistribution->offsets[d];
    int length = workDistribution->lengths[d];

    // Allocate memory on the device with plain pointers.
    int N = this->N;
    floatType *p = this->p.get();
    floatType *q = this->q.get();
    floatType *r = this->r.get();
    floatType *x = this->x;
    floatType *k = this->k;

    #pragma acc enter data async create(p[0:N], q[offset:length]) \
                                 create(r[offset:length]) \
                                 copyin(x[0:N], k[offset:length])
    switch (matrixFormat) {
    case MatrixFormatCRS:
      enterMatrixCRS(splitMatrixCRS->data[d], length);
      break;
    case MatrixFormatELL:
      enterMatrixELL(splitMatrixELL->data[d], length);
      break;
    default:
      assert(0 && "Invalid matrix format!");
    }
    if (preconditioner != PreconditionerNone) {
      floatType *z = this->z.get();
      #pragma acc enter data async create(z[offset:length])

      switch (preconditioner) {
      case PreconditionerJacobi: {
        floatType *C = jacobi->C;
        #pragma acc enter data async copyin(C[offset:length])
        break;
      }
      default:
        assert(0 && "Invalid preconditioner!");
      }
    }
  }

  waitForAllDevices();
}

static inline void exitMatrixCRS(const MatrixDataCRS &matrix, int N) {
  int *ptr = matrix.ptr;
  int nz = ptr[N];
  int *index = matrix.index;
  floatType *value = matrix.value;

  #pragma acc exit data delete(ptr[0:N+1], index[0:nz], value[0:nz])
}

static inline void exitMatrixELL(const MatrixDataELL &matrix, int N) {
  int elements = matrix.elements;
  int *length = matrix.length;
  int *index = matrix.index;
  floatType *data = matrix.data;

  #pragma acc exit data async delete(length[0:N], index[0:elements]) \
                              delete(data[0:elements])
}

void CGMultiOpenACC::doTransferFrom() {
  for (int d = 0; d < getNumberOfDevices(); d++) {
    acc_set_device_num(d, acc_get_device_type());
    int offset = workDistribution->offsets[d];
    int length = workDistribution->lengths[d];

    // Free memory on the device with plain pointers.
    int N = this->N;
    floatType *p = this->p.get();
    floatType *q = this->q.get();
    floatType *r = this->r.get();
    floatType *x = this->x;
    floatType *k = this->k;

    #pragma acc update async host(x[offset:length])
    #pragma acc exit data async delete(p[0:N], q[offset:length]) \
                                delete(r[offset:length], k[offset:length]) \
                                delete(x[0:N])
    switch (matrixFormat) {
    case MatrixFormatCRS:
      exitMatrixCRS(splitMatrixCRS->data[d], length);
      break;
    case MatrixFormatELL:
      exitMatrixELL(splitMatrixELL->data[d], length);
      break;
    default:
      assert(0 && "Invalid matrix format!");
    }
    if (preconditioner != PreconditionerNone) {
      floatType *z = this->z.get();
      #pragma acc exit data async delete(z[offset:length])

      switch (preconditioner) {
      case PreconditionerJacobi: {
        floatType *C = jacobi->C;
        #pragma acc exit data async delete(C[offset:length])
        break;
      }
      default:
        assert(0 && "Invalid preconditioner!");
      }
    }
  }

  waitForAllDevices();
}

void CGMultiOpenACC::cpy(Vector _dst, Vector _src) {
  floatType *dst = getVector(_dst);
  floatType *src = getVector(_src);

  for (int d = 0; d < getNumberOfDevices(); d++) {
    acc_set_device_num(d, acc_get_device_type());
    int offset = workDistribution->offsets[d];
    int length = workDistribution->lengths[d];

#pragma acc parallel loop async present(dst[offset:length], src[offset:length])
    for (int i = offset; i < offset + length; i++) {
      dst[i] = src[i];
    }
  }

  waitForAllDevices();
}

void CGMultiOpenACC::matvecGatherXViaHost(floatType *x) {
  // Gather x on host.
  for (int d = 0; d < getNumberOfDevices(); d++) {
    acc_set_device_num(d, acc_get_device_type());
    int offset = workDistribution->offsets[d];
    int length = workDistribution->lengths[d];

    #pragma acc update async host(x[offset:length])
  }
  waitForAllDevices();

  // Transfer x to devices.
  for (int d = 0; d < getNumberOfDevices(); d++) {
    acc_set_device_num(d, acc_get_device_type());

    for (int src = 0; src < getNumberOfDevices(); src++) {
      if (src == d) {
        // Don't transfer chunk that is already on the device.
        continue;
      }
      int offset = workDistribution->offsets[src];
      int length = workDistribution->lengths[src];

      #pragma acc update async device(x[offset:length])
    }
  }
  waitForAllDevices();
}

void CGMultiOpenACC::matvecKernelCRS(floatType *x, floatType *y) {
  int N = this->N;

  for (int d = 0; d < getNumberOfDevices(); d++) {
    acc_set_device_num(d, acc_get_device_type());
    int offset = workDistribution->offsets[d];
    int length = workDistribution->lengths[d];

    int *ptr = splitMatrixCRS->data[d].ptr;
    int nz = ptr[length];
    int *index = splitMatrixCRS->data[d].index;
    floatType *value = splitMatrixCRS->data[d].value;

#pragma acc parallel loop async gang vector present(x[0:N], y[offset:length]) \
                          present(ptr[0:length+1], index[0:nz], value[0:nz])
    for (int i = 0; i < length; i++) {
      floatType tmp = 0;
      for (int j = ptr[i]; j < ptr[i + 1]; j++) {
        tmp += value[j] * x[index[j]];
      }
      y[offset + i] = tmp;
    }
  }

  waitForAllDevices();
}

void CGMultiOpenACC::matvecKernelELL(floatType *x, floatType *y) {
  int N = this->N;

  for (int d = 0; d < getNumberOfDevices(); d++) {
    acc_set_device_num(d, acc_get_device_type());
    int offset = workDistribution->offsets[d];
    int length = workDistribution->lengths[d];

    int elements = splitMatrixELL->data[d].elements;
    int *lengthA = splitMatrixELL->data[d].length;
    int *index = splitMatrixELL->data[d].index;
    floatType *data = splitMatrixELL->data[d].data;

#pragma acc parallel loop async gang vector present(x[0:N], y[offset:length]) \
                          present(lengthA[0:length], index[0:elements]) \
                          present(data[0:elements])
    for (int i = 0; i < length; i++) {
      floatType tmp = 0;
      for (int j = 0; j < lengthA[i]; j++) {
        // long is to work-around an overflow in address calculation...
        long k = j * length + i;
        tmp += data[k] * x[index[k]];
      }
      y[offset + i] = tmp;
    }
  }

  waitForAllDevices();
}

void CGMultiOpenACC::matvecKernel(Vector _x, Vector _y) {
  floatType *x = getVector(_x);
  floatType *y = getVector(_y);

  matvecGatherXViaHost(x);

  switch (matrixFormat) {
  case MatrixFormatCRS:
    matvecKernelCRS(x, y);
    break;
  case MatrixFormatELL:
    matvecKernelELL(x, y);
    break;
  default:
    assert(0 && "Invalid matrix format!");
  }
}

void CGMultiOpenACC::axpyKernel(floatType a, Vector _x, Vector _y) {
  floatType *x = getVector(_x);
  floatType *y = getVector(_y);

  for (int d = 0; d < getNumberOfDevices(); d++) {
    acc_set_device_num(d, acc_get_device_type());
    int offset = workDistribution->offsets[d];
    int length = workDistribution->lengths[d];

#pragma acc parallel loop async present(x[offset:length], y[offset:length])
    for (int i = offset; i < offset + length; i++) {
      y[i] += a * x[i];
    }
  }

  waitForAllDevices();
}

void CGMultiOpenACC::xpayKernel(Vector _x, floatType a, Vector _y) {
  floatType *x = getVector(_x);
  floatType *y = getVector(_y);

  for (int d = 0; d < getNumberOfDevices(); d++) {
    acc_set_device_num(d, acc_get_device_type());
    int offset = workDistribution->offsets[d];
    int length = workDistribution->lengths[d];

#pragma acc parallel loop async present(x[offset:length], y[offset:length])
    for (int i = offset; i < offset + length; i++) {
      y[i] = x[i] + a * y[i];
    }
  }

  waitForAllDevices();
}

floatType CGMultiOpenACC::vectorDotKernel(Vector _a, Vector _b) {
  floatType res = 0, res_dev = 0;
  floatType *a = getVector(_a);
  floatType *b = getVector(_b);

  for (int d = 0; d < getNumberOfDevices(); d++) {
    acc_set_device_num(d, acc_get_device_type());
    int offset = workDistribution->offsets[d];
    int length = workDistribution->lengths[d];

    // FIXME: Not standard compliant! Multiple accelerators may share storage!
    #pragma acc enter data async copyin(res_dev)

#pragma acc parallel loop async present(a[offset:length], b[offset:length]) \
                                present(res_dev) reduction(+:res_dev)
    for (int i = offset; i < offset + length; i++) {
      res_dev += a[i] * b[i];
    }
  }

  // Wait for devices and reduce partial results.
  for (int d = 0; d < getNumberOfDevices(); d++) {
    acc_set_device_num(d, acc_get_device_type());
    #pragma acc wait
    #pragma acc exit data copyout(res_dev)
    res += res_dev;
  }

  return res;
}

void CGMultiOpenACC::applyPreconditionerKernelJacobi(floatType *x,
                                                     floatType *y) {
  floatType *C = jacobi->C;

  for (int d = 0; d < getNumberOfDevices(); d++) {
    acc_set_device_num(d, acc_get_device_type());
    int offset = workDistribution->offsets[d];
    int length = workDistribution->lengths[d];

#pragma acc parallel loop async present(x[offset:length], y[offset:length]) \
                                present(C[offset:length])
    for (int i = offset; i < offset + length; i++) {
      y[i] = C[i] * x[i];
    }
  }

  waitForAllDevices();
}

void CGMultiOpenACC::applyPreconditionerKernel(Vector _x, Vector _y) {
  floatType *x = getVector(_x);
  floatType *y = getVector(_y);

  switch (preconditioner) {
  case PreconditionerJacobi:
    applyPreconditionerKernelJacobi(x, y);
    break;
  default:
    assert(0 && "Invalid preconditioner!");
  }
}

CG *CG::getInstance() { return new CGMultiOpenACC; }
