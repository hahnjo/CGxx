#include <cassert>
#include <memory>

#include <openacc.h>

#include "../CG.h"
#include "../Matrix.h"
#include "../Preconditioner.h"

/// Class implementing parallel kernels with OpenACC.
class CGOpenACC : public CG {
  std::unique_ptr<floatType[]> p;
  std::unique_ptr<floatType[]> q;
  std::unique_ptr<floatType[]> r;
  std::unique_ptr<floatType[]> z;

  floatType *getVector(Vector v) {
    switch (v) {
    case VectorK:
      return k.get();
    case VectorX:
      return x.get();
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

  virtual void init(const char *matrixFile) override;

  virtual bool needsTransfer() override { return true; }
  virtual void doTransferTo() override;
  virtual void doTransferFrom() override;

  virtual void cpy(Vector _dst, Vector _src) override;

  void matvecKernelCRS(floatType *x, floatType *y);
  void matvecKernelELL(floatType *x, floatType *y);

  virtual void matvecKernel(Vector _x, Vector _y) override;
  virtual void axpyKernel(floatType a, Vector _x, Vector _y) override;
  virtual void xpayKernel(Vector _x, floatType a, Vector _y) override;
  virtual floatType vectorDotKernel(Vector _a, Vector _b) override;

  void applyPreconditionerKernelJacobi(floatType *x, floatType *y);

  virtual void applyPreconditionerKernel(Vector _x, Vector _y) override;

public:
  CGOpenACC() : CG(MatrixFormatELL, PreconditionerJacobi) {}
};

void CGOpenACC::init(const char *matrixFile) {
  // First init the device.
  acc_init(acc_get_device_type());

  CG::init(matrixFile);

  p.reset(new floatType[N]);
  q.reset(new floatType[N]);
  r.reset(new floatType[N]);
  if (preconditioner != PreconditionerNone) {
    z.reset(new floatType[N]);
  }
}

void CGOpenACC::doTransferTo() {
  // Allocate memory on the device with plain pointers.
  int N = this->N, nz = this->nz;
  floatType *p = this->p.get();
  floatType *q = this->q.get();
  floatType *r = this->r.get();
  floatType *x = this->x.get();
  floatType *k = this->k.get();

  #pragma acc enter data create(p[0:N], q[0:N], r[0:N]) copyin(x[0:N], k[0:N])
  switch (matrixFormat) {
  case MatrixFormatCRS: {
    int *ptr = matrixCRS->ptr.get();
    int *index = matrixCRS->index.get();
    floatType *value = matrixCRS->value.get();
    #pragma acc enter data copyin(ptr[0:N+1], index[0:nz], value[0:nz])
    break;
  }
  case MatrixFormatELL: {
    int elements = matrixELL->elements;
    int *length = matrixELL->length.get();
    int *index = matrixELL->index.get();
    floatType *data = matrixELL->data.get();
    #pragma acc enter data copyin(length[0:N]) \
                           copyin(index[0:elements], data[0:elements])
    break;
  }
  default:
    assert(0 && "Invalid matrix format!");
  }
  if (preconditioner != PreconditionerNone) {
    floatType *z = this->z.get();
    #pragma acc enter data create(z[0:N])

    switch (preconditioner) {
    case PreconditionerJacobi: {
      floatType *C = jacobi->C.get();
      #pragma acc enter data copyin(C[0:N])
      break;
    }
    default:
      assert(0 && "Invalid preconditioner!");
    }
  }
}

void CGOpenACC::doTransferFrom() {
  // Free memory on the device with plain pointers.
  int N = this->N, nz = this->nz;
  floatType *p = this->p.get();
  floatType *q = this->q.get();
  floatType *r = this->r.get();
  floatType *x = this->x.get();
  floatType *k = this->k.get();

  #pragma acc exit data delete(p[0:N], q[0:N], r[0:N], k[0:N]) copyout(x[0:N])
  switch (matrixFormat) {
  case MatrixFormatCRS: {
    int *ptr = matrixCRS->ptr.get();
    int *index = matrixCRS->index.get();
    floatType *value = matrixCRS->value.get();
    #pragma acc exit data delete(ptr[0:N+1], index[0:nz], value[0:nz])
    break;
  }
  case MatrixFormatELL: {
    int elements = matrixELL->elements;
    int *length = matrixELL->length.get();
    int *index = matrixELL->index.get();
    floatType *data = matrixELL->data.get();
    #pragma acc exit data delete(length[0:N]) \
                          delete(index[0:elements], data[0:elements])
    break;
  }
  default:
    assert(0 && "Invalid matrix format!");
  }
  if (preconditioner != PreconditionerNone) {
    floatType *z = this->z.get();
    #pragma acc exit data delete(z[0:N])

    switch (preconditioner) {
    case PreconditionerJacobi: {
      floatType *C = jacobi->C.get();
      #pragma acc exit data delete(C[0:N])
      break;
    }
    default:
      assert(0 && "Invalid preconditioner!");
    }
  }
}

void CGOpenACC::cpy(Vector _dst, Vector _src) {
  int N = this->N;
  floatType *dst = getVector(_dst);
  floatType *src = getVector(_src);

#pragma acc parallel loop present(dst[0:N], src[0:N])
  for (int i = 0; i < N; i++) {
    dst[i] = src[i];
  }
}

void CGOpenACC::matvecKernelCRS(floatType *x, floatType *y) {
  int N = this->N;
  int nz = this->nz;
  int *ptr = matrixCRS->ptr.get();
  int *index = matrixCRS->index.get();
  floatType *value = matrixCRS->value.get();

#pragma acc parallel loop gang vector present(x[0:N], y[0:N]) \
                          present(ptr[0:N+1], index[0:nz], value[0:nz])
  for (int i = 0; i < N; i++) {
    floatType tmp = 0;
    for (int j = ptr[i]; j < ptr[i + 1]; j++) {
      tmp += value[j] * x[index[j]];
    }
    y[i] = tmp;
  }
}

void CGOpenACC::matvecKernelELL(floatType *x, floatType *y) {
  int N = this->N;
  int elements = matrixELL->elements;
  int *length = matrixELL->length.get();
  int *index = matrixELL->index.get();
  floatType *data = matrixELL->data.get();

#pragma acc parallel loop gang vector present(x[0:N], y[0:N], length[0:N]) \
                          present(index[0:elements], data[0:elements])
  for (int i = 0; i < N; i++) {
    floatType tmp = 0;
    for (int j = 0; j < length[i]; j++) {
      // long is to work-around an overflow in address calculation...
      long k = j * N + i;
      tmp += data[k] * x[index[k]];
    }
    y[i] = tmp;
  }
}

void CGOpenACC::matvecKernel(Vector _x, Vector _y) {
  floatType *x = getVector(_x);
  floatType *y = getVector(_y);

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

void CGOpenACC::axpyKernel(floatType a, Vector _x, Vector _y) {
  int N = this->N;
  floatType *x = getVector(_x);
  floatType *y = getVector(_y);

#pragma acc parallel loop present(x[0:N], y[0:N])
  for (int i = 0; i < N; i++) {
    y[i] += a * x[i];
  }
}

void CGOpenACC::xpayKernel(Vector _x, floatType a, Vector _y) {
  int N = this->N;
  floatType *x = getVector(_x);
  floatType *y = getVector(_y);

#pragma acc parallel loop present(x[0:N], y[0:N])
  for (int i = 0; i < N; i++) {
    y[i] = x[i] + a * y[i];
  }
}

floatType CGOpenACC::vectorDotKernel(Vector _a, Vector _b) {
  int N = this->N;
  floatType res = 0;
  floatType *a = getVector(_a);
  floatType *b = getVector(_b);

#pragma acc parallel loop reduction(+:res) copy(res) present(a[0:N], b[0:N])
  for (int i = 0; i < N; i++) {
    res += a[i] * b[i];
  }

  return res;
}

void CGOpenACC::applyPreconditionerKernelJacobi(floatType *x, floatType *y) {
  int N = this->N;
  floatType *C = jacobi->C.get();

#pragma acc parallel loop present(x[0:N], y[0:N], C[0:N])
  for (int i = 0; i < N; i++) {
    y[i] = C[i] * x[i];
  }
}

void CGOpenACC::applyPreconditionerKernel(Vector _x, Vector _y) {
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

CG *CG::getInstance() { return new CGOpenACC; }
