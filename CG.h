#ifndef CG_H
#define CG_H

#include <cassert>
#include <chrono>
#include <memory>

#include "Matrix.h"
#include "Preconditioner.h"
#include "WorkDistribution.h"
#include "def.h"

/// @brief The base class implementing the conjugate gradients method.
///
/// It is used to solve the equation system Ax = k. A is a sparse matrix
/// stored either COO, CRS or ELLPACK format.
class CG {
public:
  /// Different vectors used to solve the equation system.
  enum Vector {
    /// LHS of the equation system.
    VectorK,
    /// Computed solution of the equation system.
    VectorX,
    /// Temporary vector for the search direction.
    VectorP,
    /// Temporary vector holding the result of the matrix vector multiplication.
    VectorQ,
    /// Temporary vector for the residual.
    VectorR,
    /// Temporary vector in use with the preconditioner.
    VectorZ,
  };

  /// Different formats used to store the sparse matrix.
  enum MatrixFormat {
    /// %Matrix is represented by CG#matrixCOO.
    MatrixFormatCOO,
    /// %Matrix is represented by either CG#matrixCRS, CG#splitMatrixCRS, or
    /// CG#partitionedMatrixCRS.
    MatrixFormatCRS,
    /// %Matrix is represented by either CG#matrixELL, CG#splitMatrixELL, or
    /// CG#partitionedMatrixELL.
    MatrixFormatELL,
  };

  /// Different preconditioners to use.
  enum Preconditioner {
    /// Use no preconditioner.
    PreconditionerNone,
    /// Use a Jacobi preconditioner.
    PreconditionerJacobi,
  };

private:
  int iteration;
  int maxIterations = 1000;

  floatType residual;
  floatType tolerance = 1e-9;

  /// Struct holding timing information for IO, converting, the total solve time
  /// and for each kernel.
  struct Timing {
    using clock = std::chrono::steady_clock;
    using duration = std::chrono::duration<double>;

    duration io{0};
    duration converting{0};
    duration transferTo{0};
    duration transferFrom{0};

    duration solve{0};
    duration matvec{0};
    duration axpy{0};
    duration xpay{0};
    duration vectorDot{0};
    duration preconditioner{0};
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

  void applyPreconditioner(Vector x, Vector y) {
    time_point start = now();
    applyPreconditionerKernel(x, y);
    timing.preconditioner += now() - start;
  }

protected:
  /// Dimension of the matrix.
  int N;
  /// Nonzeros in the matrix.
  int nz;

  /// How the work is distributed into multiple chunks.
  std::unique_ptr<WorkDistribution> workDistribution;
  /// Whether to overlap the gather with some computation of matvec().
  bool overlappedGather = false;

  /// Format to store the matrix.
  MatrixFormat matrixFormat;
  /// Matrix in cooridinate format.
  std::unique_ptr<MatrixCOO> matrixCOO;
  /// Matrix in CRS format.
  std::unique_ptr<MatrixCRS> matrixCRS;
  /// Matrix in ELLPACK format.
  std::unique_ptr<MatrixELL> matrixELL;

  /// Matrix in CRS format, split for #workDistribution.
  std::unique_ptr<SplitMatrixCRS> splitMatrixCRS;
  /// Matrix in ELLPACK format, split for #workDistribution.
  std::unique_ptr<SplitMatrixELL> splitMatrixELL;

  /// Matrix in CRS format, partitioned for #workDistribution.
  std::unique_ptr<PartitionedMatrixCRS> partitionedMatrixCRS;
  /// Matrix in ELLPACK format, partitioned for #workDistribution.
  std::unique_ptr<PartitionedMatrixELL> partitionedMatrixELL;

  /// The preconditioner to use.
  Preconditioner preconditioner;
  /// Jacobi preconditioner.
  std::unique_ptr<Jacobi> jacobi;

  /// #VectorK
  floatType *k = nullptr;
  /// #VectorX
  floatType *x = nullptr;

  /// Construct a new object with a \a defaultMatrixFormat to store tha matrix
  /// and a \a defaultPreconditioner to use.
  CG(MatrixFormat defaultMatrixFormat,
     Preconditioner defaultPreconditioner = PreconditionerNone,
     bool overlappedGather = false)
      : overlappedGather(overlappedGather), matrixFormat(defaultMatrixFormat),
        preconditioner(defaultPreconditioner) {}

  /// @return \a true if this implementation supports \a format to store the matrix.
  virtual bool supportsMatrixFormat(MatrixFormat format) = 0;
  /// @return \a true if this implementation supports the \a preconditioner.
  virtual bool supportsPreconditioner(Preconditioner preconditioner) {
    return false;
  }

  /// @return the number of chunks that the work should be split into, or -1
  /// if no work distributition is necessary.
  virtual int getNumberOfChunks() { return -1; }
  /// @return \a true if this implementation supports overlapping the gather
  /// with some computation of matvec().
  virtual bool supportsOverlappedGather() { return false; }

  /// Convert to MatrixCRS.
  virtual void convertToMatrixCRS() {
    matrixCRS.reset(new MatrixCRS(*matrixCOO));
  }
  /// Convert to MatrixELL.
  virtual void convertToMatrixELL() {
    matrixELL.reset(new MatrixELL(*matrixCOO));
  }
  /// Convert to SplitMatrixCRS.
  virtual void convertToSplitMatrixCRS() {
    splitMatrixCRS.reset(new SplitMatrixCRS(*matrixCOO, *workDistribution));
  }
  /// Convert to SplitMatrixELL.
  virtual void convertToSplitMatrixELL() {
    splitMatrixELL.reset(new SplitMatrixELL(*matrixCOO, *workDistribution));
  }
  /// Convert to PartitionedMatrixCRS.
  virtual void convertToPartitionedMatrixCRS() {
    partitionedMatrixCRS.reset(
        new PartitionedMatrixCRS(*matrixCOO, *workDistribution));
  }
  /// Convert to PartitionedMatrixELL.
  virtual void convertToPartitionedMatrixELL() {
    partitionedMatrixELL.reset(
        new PartitionedMatrixELL(*matrixCOO, *workDistribution));
  }

  /// Initialize the Jacobi preconditioner.
  virtual void initJacobi() { jacobi.reset(new Jacobi(*matrixCOO)); }

  /// Allocate #k.
  virtual void allocateK() { k = new floatType[N]; }
  /// Deallocate #k.
  virtual void deallocateK() { delete[] k; }
  /// Allocate #x.
  virtual void allocateX() { x = new floatType[N]; }
  /// Deallocate #x.
  virtual void deallocateX() { delete[] x; }

  /// Do transfer data before calling #solve().
  virtual void doTransferTo() {}
  /// Do transfer data after calling #solve().
  virtual void doTransferFrom() {}

  /// Copy vector \a _src to \a _dst.
  virtual void cpy(Vector _dst, Vector _src) = 0;
  /// \a _y = A * \a _x.
  virtual void matvecKernel(Vector _x, Vector _y) = 0;
  /// \a _y = \a a * \a _x + \a _y.
  virtual void axpyKernel(floatType a, Vector _x, Vector _y) = 0;
  /// \a _y = \a _y + \a a * \a _y.
  virtual void xpayKernel(Vector _x, floatType a, Vector _y) = 0;
  /// @return vector dot product <\a _a, \a _b>
  virtual floatType vectorDotKernel(Vector _a, Vector _b) = 0;

  /// \a _y = B * \a _x
  virtual void applyPreconditionerKernel(Vector _x, Vector _y) {
    assert(0 && "Preconditioner not implemented!");
  }

  /// Print \a label (padded to a constant number of characters) and \a value.
  static void printPadded(const char *label, const std::string &value);

public:
  /// Parse and validate environment variables.
  virtual void parseEnvironment();
  /// Init data by reading matrix from \a matrixFile.
  virtual void init(const char *matrixFile);

  /// @return true if this implementation needs to transfer data for solving.
  virtual bool needsTransfer() { return false; }
  /// Transfer data before calling #solve().
  void transferTo() {
    auto start = now();
    doTransferTo();
    timing.transferTo = now() - start;
  }

  /// Solve sparse equation system.
  void solve();

  /// Transfer data after calling #solve().
  void transferFrom() {
    auto start = now();
    doTransferFrom();
    timing.transferFrom = now() - start;
  }

  /// Print summary after system has been solved.
  virtual void printSummary();

  /// Cleanup allocated memory.
  virtual void cleanup() {
    // Uses virtual methods and therefore cannot be done in destructor.
    deallocateK();
    deallocateX();
  }

  /// @return new instance of a CG implementation.
  static CG *getInstance();
};

#endif
