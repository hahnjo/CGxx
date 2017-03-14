#ifndef CG_H
#define CG_H

#include <chrono>
#include <memory>

#include "Matrix.h"
#include "def.h"

class CG {
public:
  enum Vector {
    VectorK,
    VectorX,
    VectorP,
    VectorQ,
    VectorR,
  };

  enum MatrixFormat {
    MatrixFormatCOO,
    MatrixFormatCRS,
    MatrixFormatELL,
  };

private:
  int iteration;
  int maxIterations = 1000;

  floatType residual;
  floatType tolerance = 1e-9;

  struct Timing {
    using clock = std::chrono::steady_clock;
    using duration = std::chrono::duration<double>;

    duration total;
    duration matvec;
    duration axpy;
    duration xpay;
    duration vectorDot;
  };
  Timing timing;

  using time_point = Timing::clock::time_point;
  time_point now() const { return Timing::clock::now(); }

  void matvec(Vector in, Vector out) {
    time_point start = now();
    matvecKernel(in, out);
    timing.matvec += now() - start;
  }

  void axpy(floatType a, Vector x, Vector y) {
    time_point start = now();
    axpyKernel(a, x, y);
    timing.axpy += now() - start;
  }

  void xpay(Vector x, floatType a, Vector y) {
    time_point start = now();
    xpayKernel(x, a, y);
    timing.xpay += now() - start;
  }

  floatType vectorDot(Vector a, Vector b) {
    time_point start = now();
    floatType res = vectorDotKernel(a, b);
    timing.vectorDot += now() - start;

    return res;
  }

protected:
  int N;
  int nz;

  MatrixFormat matrixFormat;
  std::unique_ptr<MatrixCOO> matrixCOO;
  std::unique_ptr<MatrixCRS> matrixCRS;
  std::unique_ptr<MatrixELL> matrixELL;

  std::unique_ptr<floatType[]> k;
  std::unique_ptr<floatType[]> x;

  CG(MatrixFormat defaultMatrixFormat) : matrixFormat(defaultMatrixFormat) {}

  virtual bool supportsMatrixFormat(MatrixFormat format) = 0;

  virtual void allocateMatrixCRS();
  virtual void allocateMatrixELL();

  virtual void allocateK();
  virtual void allocateX();

  virtual void cpy(Vector _dst, Vector _src) = 0;
  virtual void matvecKernel(Vector _x, Vector _y) = 0;
  virtual void axpyKernel(floatType a, Vector _x, Vector _y) = 0;
  virtual void xpayKernel(Vector _x, floatType a, Vector _y) = 0;
  virtual floatType vectorDotKernel(Vector _a, Vector _b) = 0;

  static void printPadded(const char *label, const std::string &value);

public:
  virtual void parseEnvironment();
  virtual void init(const char *matrixFile);

  void solve();

  virtual void printSummary();

  static CG *getInstance();
};

#endif
