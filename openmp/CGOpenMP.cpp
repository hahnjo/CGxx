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

#include <cassert>
#include <memory>

#include "../CG.h"
#include "../Matrix.h"
#include "../Preconditioner.h"

/// Class implementing parallel kernels with OpenMP.
class CGOpenMP : public CG {
  struct MatrixCRSOpenMP : MatrixCRS {
    MatrixCRSOpenMP(const MatrixCOO &coo) : MatrixCRS(coo) {}

    virtual void allocatePtr(int rows) override;
    virtual void allocateIndexAndValue(int values) override;
  };
  struct MatrixELLOpenMP : MatrixELL {
    MatrixELLOpenMP(const MatrixCOO &coo) : MatrixELL(coo) {}

    virtual void allocateLength(int rows) override;
    virtual void allocateIndexAndData() override;
  };
  struct JacobiOpenMP : Jacobi {
    JacobiOpenMP(const MatrixCOO &coo) : Jacobi(coo) {}

    virtual void allocateC(int N) override;
  };

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

  virtual void init(const char *matrixFile) override;

  virtual void convertToMatrixCRS() override {
    matrixCRS.reset(new MatrixCRSOpenMP(*matrixCOO));
  }
  virtual void convertToMatrixELL() override {
    matrixELL.reset(new MatrixELLOpenMP(*matrixCOO));
  }

  virtual void initJacobi() override {
    jacobi.reset(new JacobiOpenMP(*matrixCOO));
  }

  virtual void allocateK() override;
  virtual void allocateX() override;

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
  CGOpenMP() : CG(MatrixFormatCRS, PreconditionerJacobi) {}
};

void CGOpenMP::MatrixCRSOpenMP::allocatePtr(int rows) {
  MatrixCRS::allocatePtr(rows);

#pragma omp parallel for
  for (int i = 0; i < rows + 1; i++) {
    ptr[i] = 0;
  }
}

void CGOpenMP::MatrixCRSOpenMP::allocateIndexAndValue(int values) {
  MatrixCRS::allocateIndexAndValue(values);

#pragma omp parallel for
  for (int i = 0; i < N; i++) {
    for (int j = ptr[i]; j < ptr[i + 1]; j++) {
      index[j] = 0;
      value[j] = 0.0;
    }
  }
}

void CGOpenMP::MatrixELLOpenMP::allocateLength(int rows) {
  MatrixELL::allocateLength(rows);

#pragma omp parallel for
  for (int i = 0; i < rows; i++) {
    length[i] = 0;
  }
}

void CGOpenMP::MatrixELLOpenMP::allocateIndexAndData() {
  MatrixELL::allocateIndexAndData();

#pragma omp parallel for
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < length[i]; j++) {
      int k = j * N + i;
      index[k] = 0;
      data[k] = 0.0;
    }
  }
}

void CGOpenMP::JacobiOpenMP::allocateC(int N) {
  Jacobi::allocateC(N);

#pragma omp parallel for
  for (int i = 0; i < N; i++) {
    C[i] = 0.0;
  }
}

void CGOpenMP::init(const char *matrixFile) {
  CG::init(matrixFile);

  p.reset(new floatType[N]);
  q.reset(new floatType[N]);
  r.reset(new floatType[N]);
  if (preconditioner != PreconditionerNone) {
    z.reset(new floatType[N]);
  }

#pragma omp parallel for
  for (int i = 0; i < N; i++) {
    p[i] = 0.0;
    q[i] = 0.0;
    r[i] = 0.0;
    if (preconditioner != PreconditionerNone) {
      z[i] = 0.0;
    }
  }
}

void CGOpenMP::allocateK() {
  CG::allocateK();

#pragma omp parallel for
  for (int i = 0; i < N; i++) {
    k[i] = 0.0;
  }
}

void CGOpenMP::allocateX() {
  CG::allocateX();

#pragma omp parallel for
  for (int i = 0; i < N; i++) {
    x[i] = 0.0;
  }
}

void CGOpenMP::cpy(Vector _dst, Vector _src) {
  floatType *dst = getVector(_dst);
  floatType *src = getVector(_src);

#pragma omp parallel for
  for (int i = 0; i < N; i++) {
    dst[i] = src[i];
  }
}

void CGOpenMP::matvecKernelCRS(floatType *x, floatType *y) {
#pragma omp parallel for
  for (int i = 0; i < N; i++) {
    floatType tmp = 0;
    for (int j = matrixCRS->ptr[i]; j < matrixCRS->ptr[i + 1]; j++) {
      tmp += matrixCRS->value[j] * x[matrixCRS->index[j]];
    }
    y[i] = tmp;
  }
}

void CGOpenMP::matvecKernelELL(floatType *x, floatType *y) {
#pragma omp parallel for
  for (int i = 0; i < N; i++) {
    floatType tmp = 0;
    for (int j = 0; j < matrixELL->length[i]; j++) {
      int k = j * N + i;
      tmp += matrixELL->data[k] * x[matrixELL->index[k]];
    }
    y[i] = tmp;
  }
}

void CGOpenMP::matvecKernel(Vector _x, Vector _y) {
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

void CGOpenMP::axpyKernel(floatType a, Vector _x, Vector _y) {
  floatType *x = getVector(_x);
  floatType *y = getVector(_y);

#pragma omp parallel for
  for (int i = 0; i < N; i++) {
    y[i] += a * x[i];
  }
}

void CGOpenMP::xpayKernel(Vector _x, floatType a, Vector _y) {
  floatType *x = getVector(_x);
  floatType *y = getVector(_y);

#pragma omp parallel for
  for (int i = 0; i < N; i++) {
    y[i] = x[i] + a * y[i];
  }
}

floatType CGOpenMP::vectorDotKernel(Vector _a, Vector _b) {
  floatType res = 0;
  floatType *a = getVector(_a);
  floatType *b = getVector(_b);

#pragma omp parallel for reduction(+:res)
  for (int i = 0; i < N; i++) {
    res += a[i] * b[i];
  }

  return res;
}

void CGOpenMP::applyPreconditionerKernelJacobi(floatType *x, floatType *y) {
#pragma omp parallel for
  for (int i = 0; i < N; i++) {
    y[i] = jacobi->C[i] * x[i];
  }
}

void CGOpenMP::applyPreconditionerKernel(Vector _x, Vector _y) {
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

CG *CG::getInstance() { return new CGOpenMP; }
