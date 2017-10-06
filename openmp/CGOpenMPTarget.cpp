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

/// Class implementing parallel kernels with OpenMP target directives.
class CGOpenMPTarget : public CG {
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
  CGOpenMPTarget() : CG(MatrixFormatELL, PreconditionerJacobi) {}
};

void CGOpenMPTarget::init(const char *matrixFile) {
  // Init the device with a simple target region.
  #pragma omp target
  { }

  CG::init(matrixFile);

  p.reset(new floatType[N]);
  q.reset(new floatType[N]);
  r.reset(new floatType[N]);
  if (preconditioner != PreconditionerNone) {
    z.reset(new floatType[N]);
  }
}

void CGOpenMPTarget::doTransferTo() {
  // Allocate memory on the device with plain pointers.
  int N = this->N, nz = this->nz;
  floatType *p = this->p.get();
  floatType *q = this->q.get();
  floatType *r = this->r.get();
  floatType *x = this->x;
  floatType *k = this->k;

  #pragma omp target enter data map(alloc: p[0:N], q[0:N], r[0:N]) \
                                map(to: x[0:N], k[0:N])
  switch (matrixFormat) {
  case MatrixFormatCRS: {
    int *ptr = matrixCRS->ptr;
    int *index = matrixCRS->index;
    floatType *value = matrixCRS->value;
    #pragma omp target enter data map(to: ptr[0:N+1], index[0:nz], value[0:nz])
    break;
  }
  case MatrixFormatELL: {
    int elements = matrixELL->elements;
    int *length = matrixELL->length;
    int *index = matrixELL->index;
    floatType *data = matrixELL->data;
    #pragma omp target enter data map(to: length[0:N]) \
                                  map(to: index[0:elements], data[0:elements])
    break;
  }
  default:
    assert(0 && "Invalid matrix format!");
  }
  if (preconditioner != PreconditionerNone) {
    floatType *z = this->z.get();
    #pragma omp target enter data map(alloc: z[0:N])

    switch (preconditioner) {
    case PreconditionerJacobi: {
      floatType *C = jacobi->C;
      #pragma omp target enter data map(to: C[0:N])
      break;
    }
    default:
      assert(0 && "Invalid preconditioner!");
    }
  }
}

void CGOpenMPTarget::doTransferFrom() {
  // Free memory on the device with plain pointers.
  int N = this->N, nz = this->nz;
  floatType *p = this->p.get();
  floatType *q = this->q.get();
  floatType *r = this->r.get();
  floatType *x = this->x;
  floatType *k = this->k;

  #pragma omp target exit data map(release: p[0:N], q[0:N], r[0:N], k[0:N]) \
                               map(from: x[0:N])
  switch (matrixFormat) {
  case MatrixFormatCRS: {
    int *ptr = matrixCRS->ptr;
    int *index = matrixCRS->index;
    floatType *value = matrixCRS->value;
    #pragma omp target exit data map(release: ptr[0:N+1]) \
                                 map(release: index[0:nz], value[0:nz])
    break;
  }
  case MatrixFormatELL: {
    int elements = matrixELL->elements;
    int *length = matrixELL->length;
    int *index = matrixELL->index;
    floatType *data = matrixELL->data;
    #pragma omp target exit data map(release: length[0:N]) \
                                 map(release: index[0:elements]) \
                                 map(release: data[0:elements])
    break;
  }
  default:
    assert(0 && "Invalid matrix format!");
  }
  if (preconditioner != PreconditionerNone) {
    floatType *z = this->z.get();
    #pragma omp target exit data map(release: z[0:N])

    switch (preconditioner) {
    case PreconditionerJacobi: {
      floatType *C = jacobi->C;
      #pragma omp target exit data map(release: C[0:N])
      break;
    }
    default:
      assert(0 && "Invalid preconditioner!");
    }
  }
}

void CGOpenMPTarget::cpy(Vector _dst, Vector _src) {
  int N = this->N;
  floatType *dst = getVector(_dst);
  floatType *src = getVector(_src);

#pragma omp target teams distribute parallel for simd map(dst[0:N], src[0:N])
  for (int i = 0; i < N; i++) {
    dst[i] = src[i];
  }
}

void CGOpenMPTarget::matvecKernelCRS(floatType *x, floatType *y) {
  int N = this->N;
  int nz = this->nz;
  int *ptr = matrixCRS->ptr;
  int *index = matrixCRS->index;
  floatType *value = matrixCRS->value;

#pragma omp target teams distribute parallel for map(x[0:N], y[0:N]) \
                   map(ptr[0:N+1], index[0:nz], value[0:nz])
  for (int i = 0; i < N; i++) {
    floatType tmp = 0;
    #pragma omp simd reduction(+:tmp)
    for (int j = ptr[i]; j < ptr[i + 1]; j++) {
      tmp += value[j] * x[index[j]];
    }
    y[i] = tmp;
  }
}

void CGOpenMPTarget::matvecKernelELL(floatType *x, floatType *y) {
  int N = this->N;
  int elements = matrixELL->elements;
  int *length = matrixELL->length;
  int *index = matrixELL->index;
  floatType *data = matrixELL->data;

#pragma omp target teams distribute parallel for simd map(x[0:N], y[0:N]) \
                   map(length[0:N], index[0:elements], data[0:elements])
  for (int i = 0; i < N; i++) {
    floatType tmp = 0;
    #pragma unroll 1
    for (int j = 0; j < length[i]; j++) {
      int k = j * N + i;
      tmp += data[k] * x[index[k]];
    }
    y[i] = tmp;
  }
}

void CGOpenMPTarget::matvecKernel(Vector _x, Vector _y) {
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

void CGOpenMPTarget::axpyKernel(floatType a, Vector _x, Vector _y) {
  int N = this->N;
  floatType *x = getVector(_x);
  floatType *y = getVector(_y);

#pragma omp target teams distribute parallel for simd map(x[0:N], y[0:N])
  for (int i = 0; i < N; i++) {
    y[i] += a * x[i];
  }
}

void CGOpenMPTarget::xpayKernel(Vector _x, floatType a, Vector _y) {
  int N = this->N;
  floatType *x = getVector(_x);
  floatType *y = getVector(_y);

#pragma omp target teams distribute parallel for simd map(x[0:N], y[0:N])
  for (int i = 0; i < N; i++) {
    y[i] = x[i] + a * y[i];
  }
}

floatType CGOpenMPTarget::vectorDotKernel(Vector _a, Vector _b) {
  int N = this->N;
  floatType res = 0;
  floatType *a = getVector(_a);
  floatType *b = getVector(_b);

#pragma omp target teams distribute parallel for simd reduction(+:res) \
                   map(res, a[0:N], b[0:N])
  for (int i = 0; i < N; i++) {
    res += a[i] * b[i];
  }

  return res;
}

void CGOpenMPTarget::applyPreconditionerKernelJacobi(floatType *x,
                                                     floatType *y) {
  int N = this->N;
  floatType *C = jacobi->C;

#pragma omp target teams distribute parallel for simd \
                   map(x[0:N], y[0:N], C[0:N])
  for (int i = 0; i < N; i++) {
    y[i] = C[i] * x[i];
  }
}

void CGOpenMPTarget::applyPreconditionerKernel(Vector _x, Vector _y) {
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

CG *CG::getInstance() { return new CGOpenMPTarget; }
