#include <cassert>
#include <cstring>
#include <memory>

#include "../CG.h"

class SerialCG : public CG {
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

  virtual void init(const char *matrixFile);

  virtual void cpy(Vector _dst, Vector _src) override;
  virtual void matvecKernel(Vector _x, Vector _y) override;
  virtual void axpyKernel(floatType a, Vector _x, Vector _y) override;
  virtual void xpayKernel(Vector _x, floatType a, Vector _y) override;
  virtual floatType vectorDotKernel(Vector _a, Vector _b) override;
};

void SerialCG::init(const char *matrixFile) {
  CG::init(matrixFile);

  p.reset(new floatType[N]);
  q.reset(new floatType[N]);
  r.reset(new floatType[N]);
}

void SerialCG::cpy(Vector _dst, Vector _src) {
  std::memcpy(getVector(_dst), getVector(_src), sizeof(floatType) * N);
}

void SerialCG::matvecKernel(Vector _x, Vector _y) {
  floatType *x = getVector(_x);
  floatType *y = getVector(_y);

  std::memset(y, 0, sizeof(floatType) * N);

  for (int i = 0; i < nz; i++) {
    y[matrixCOO->I[i]] += matrixCOO->V[i] * x[matrixCOO->J[i]];
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

CG *CG::getInstance() { return new SerialCG; }
