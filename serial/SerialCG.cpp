// SPDX-License-Identifier:	GPL-3.0-or-later

#include <cassert>
#include <cstring>
#include <memory>

#include "../CG.h"

/// Class imlementing serial kernels.
class SerialCG : public CG {
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
    return format == MatrixFormatCOO || format == MatrixFormatCRS ||
           format == MatrixFormatELL;
  }
  virtual bool supportsPreconditioner(Preconditioner preconditioner) override {
    return preconditioner == PreconditionerJacobi;
  }

  virtual void init(const char *matrixFile) override;

  virtual void cpy(Vector _dst, Vector _src) override;

  void matvecKernelCOO(floatType *x, floatType *y);
  void matvecKernelCRS(floatType *x, floatType *y);
  void matvecKernelELL(floatType *x, floatType *y);

  virtual void matvecKernel(Vector _x, Vector _y) override;
  virtual void axpyKernel(floatType a, Vector _x, Vector _y) override;
  virtual void xpayKernel(Vector _x, floatType a, Vector _y) override;
  virtual floatType vectorDotKernel(Vector _a, Vector _b) override;

  void applyPreconditionerKernelJacobi(floatType *x, floatType *y);

  virtual void applyPreconditionerKernel(Vector _x, Vector _y) override;

public:
  SerialCG() : CG(MatrixFormatCRS, PreconditionerJacobi) {}
};

void SerialCG::init(const char *matrixFile) {
  CG::init(matrixFile);

  p.reset(new floatType[N]);
  q.reset(new floatType[N]);
  r.reset(new floatType[N]);
  if (preconditioner != PreconditionerNone) {
    z.reset(new floatType[N]);
  }
}

void SerialCG::cpy(Vector _dst, Vector _src) {
  std::memcpy(getVector(_dst), getVector(_src), sizeof(floatType) * N);
}

void SerialCG::matvecKernelCOO(floatType *x, floatType *y) {
  std::memset(y, 0, sizeof(floatType) * N);

  for (int i = 0; i < nz; i++) {
    y[matrixCOO->I[i]] += matrixCOO->V[i] * x[matrixCOO->J[i]];
  }
}

void SerialCG::matvecKernelCRS(floatType *x, floatType *y) {
  for (int i = 0; i < N; i++) {
    floatType tmp = 0;
    for (int j = matrixCRS->ptr[i]; j < matrixCRS->ptr[i + 1]; j++) {
      tmp += matrixCRS->value[j] * x[matrixCRS->index[j]];
    }
    y[i] = tmp;
  }
}

void SerialCG::matvecKernelELL(floatType *x, floatType *y) {
  for (int i = 0; i < N; i++) {
    floatType tmp = 0;
    for (int j = 0; j < matrixELL->length[i]; j++) {
      int k = j * N + i;
      tmp += matrixELL->data[k] * x[matrixELL->index[k]];
    }
    y[i] = tmp;
  }
}

void SerialCG::matvecKernel(Vector _x, Vector _y) {
  floatType *x = getVector(_x);
  floatType *y = getVector(_y);

  switch (matrixFormat) {
  case MatrixFormatCOO:
    matvecKernelCOO(x, y);
    break;
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

void SerialCG::axpyKernel(floatType a, Vector _x, Vector _y) {
  floatType *x = getVector(_x);
  floatType *y = getVector(_y);

  for (int i = 0; i < N; i++) {
    y[i] += a * x[i];
  }
}

void SerialCG::xpayKernel(Vector _x, floatType a, Vector _y) {
  floatType *x = getVector(_x);
  floatType *y = getVector(_y);

  for (int i = 0; i < N; i++) {
    y[i] = x[i] + a * y[i];
  }
}

floatType SerialCG::vectorDotKernel(Vector _a, Vector _b) {
  floatType res = 0;
  floatType *a = getVector(_a);
  floatType *b = getVector(_b);

  for (int i = 0; i < N; i++) {
    res += a[i] * b[i];
  }

  return res;
}

void SerialCG::applyPreconditionerKernelJacobi(floatType *x, floatType *y) {
  for (int i = 0; i < N; i++) {
    y[i] = jacobi->C[i] * x[i];
  }
}

void SerialCG::applyPreconditionerKernel(Vector _x, Vector _y) {
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

CG *CG::getInstance() { return new SerialCG; }
