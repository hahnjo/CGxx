#include <cassert>
#include <memory>

#include "../CG.h"
#include "../Matrix.h"

class CGOpenMP : public CG {
  struct MatrixCRSOpenMP : MatrixCRS {
    MatrixCRSOpenMP(const MatrixCOO &coo) : MatrixCRS(coo) {}

    virtual void allocatePtr() override;
    virtual void allocateIndexAndValue() override;
  };
  struct MatrixELLOpenMP : MatrixELL {
    MatrixELLOpenMP(const MatrixCOO &coo) : MatrixELL(coo) {}

    virtual void allocateLength() override;
    virtual void allocateIndexAndData() override;
  };

  std::unique_ptr<floatType[]> p;
  std::unique_ptr<floatType[]> q;
  std::unique_ptr<floatType[]> r;

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
    }
    assert(0 && "Invalid value of v!");
    return nullptr;
  }

  virtual bool supportsMatrixFormat(MatrixFormat format) override {
    return format == MatrixFormatCRS || format == MatrixFormatELL;
  }

  virtual void init(const char *matrixFile) override;

  virtual void convertToMatrixCRS() override {
    matrixCRS.reset(new MatrixCRSOpenMP(*matrixCOO));
  }
  virtual void convertToMatrixELL() override {
    matrixELL.reset(new MatrixELLOpenMP(*matrixCOO));
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

public:
  CGOpenMP() : CG(MatrixFormatCRS) {}
};

void CGOpenMP::MatrixCRSOpenMP::allocatePtr() {
  MatrixCRS::allocatePtr();

#pragma omp parallel for
  for (int i = 0; i < N + 1; i++) {
    ptr[i] = 0;
  }
}

void CGOpenMP::MatrixCRSOpenMP::allocateIndexAndValue() {
  MatrixCRS::allocateIndexAndValue();

#pragma omp parallel for
  for (int i = 0; i < N; i++) {
    for (int j = ptr[i]; j < ptr[i + 1]; j++) {
      index[j] = 0;
      value[j] = 0.0;
    }
  }
}

void CGOpenMP::MatrixELLOpenMP::allocateLength() {
  MatrixELL::allocateLength();

#pragma omp parallel for
  for (int i = 0; i < N; i++) {
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

void CGOpenMP::init(const char *matrixFile) {
  CG::init(matrixFile);

  p.reset(new floatType[N]);
  q.reset(new floatType[N]);
  r.reset(new floatType[N]);

#pragma omp parallel for
  for (int i = 0; i < N; i++) {
    p[i] = 0.0;
    q[i] = 0.0;
    r[i] = 0.0;
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

CG *CG::getInstance() { return new CGOpenMP; }
