// SPDX-License-Identifier:	GPL-3.0-or-later

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>

#include "CG.h"
#include "Matrix.h"
#include "WorkDistribution.h"

const char *CG_MAX_ITER = "CG_MAX_ITER";
const char *CG_TOLERANCE = "CG_TOLERANCE";
const char *CG_CHECK_TOLERANCE = "CG_CHECK_TOLERANCE";

const char *CG_MATRIX_FORMAT = "CG_MATRIX_FORMAT";
const char *CG_MATRIX_FORMAT_COO = "COO";
const char *CG_MATRIX_FORMAT_CRS = "CRS";
const char *CG_MATRIX_FORMAT_ELL = "ELL";

const char *CG_PRECONDITIONER = "CG_PRECONDITIONER";
const char *CG_PRECONDITIONER_NONE = "none";
const char *CG_PRECONDITIONER_JACOBI = "jacobi";

const char *CG_WORK_DISTRIBUTION = "CG_WORK_DISTRIBUTION";
const char *CG_WORK_DISTRIBUTION_BY_ROW = "row";
const char *CG_WORK_DISTRIBUTION_BY_NZ = "nz";

const char *CG_OVERLAPPED_GATHER = "CG_OVERLAPPED_GATHER";

void CG::parseEnvironment() {
  const char *env;
  char *endptr;

  env = std::getenv(CG_MAX_ITER);
  if (env != NULL && *env != 0) {
    errno = 0;
    int maxIterations = strtol(env, &endptr, 0);
    if (errno == 0 && *endptr == 0 && maxIterations > 0) {
      this->maxIterations = maxIterations;
    } else {
      std::cerr << "Invalid value for " << CG_MAX_ITER << "!" << std::endl;
      std::exit(1);
    }
  }

  env = std::getenv(CG_TOLERANCE);
  if (env != NULL && *env != 0) {
    errno = 0;
    floatType tolerance = strtod(env, &endptr);
    if (errno == 0 && *endptr == 0 && tolerance > 0) {
      this->tolerance = tolerance;
    } else {
      std::cerr << "Invalid value for " << CG_TOLERANCE << "!" << std::endl;
      std::exit(1);
    }
  }

  env = std::getenv(CG_CHECK_TOLERANCE);
  if (env != NULL && *env != 0) {
    errno = 0;
    floatType checkTolerance = strtod(env, &endptr);
    if (errno == 0 && *endptr == 0 && tolerance > 0) {
      this->checkTolerance = checkTolerance;
    } else {
      std::cerr << "Invalid value for " << CG_CHECK_TOLERANCE << "!"
                << std::endl;
      std::exit(1);
    }
  }

  env = std::getenv(CG_MATRIX_FORMAT);
  if (env != NULL && *env != 0) {
    std::string upper(env);
    std::transform(upper.begin(), upper.end(), upper.begin(), ::toupper);

    if (upper == CG_MATRIX_FORMAT_COO) {
      matrixFormat = MatrixFormatCOO;
    } else if (upper == CG_MATRIX_FORMAT_CRS) {
      matrixFormat = MatrixFormatCRS;
    } else if (upper == CG_MATRIX_FORMAT_ELL) {
      matrixFormat = MatrixFormatELL;
    } else {
      std::cerr << "Invalid value for " << CG_MATRIX_FORMAT << "! ("
                << CG_MATRIX_FORMAT_COO << ", " << CG_MATRIX_FORMAT_CRS
                << ", or " << CG_MATRIX_FORMAT_ELL << ")" << std::endl;
      std::exit(1);
    }

    if (!supportsMatrixFormat(matrixFormat)) {
      std::cerr << "No support for this matrix format!" << std::endl;
      std::exit(1);
    }
  }

  env = std::getenv(CG_PRECONDITIONER);
  if (env != NULL && *env != 0) {
    std::string lower(env);
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

    if (lower == CG_PRECONDITIONER_NONE) {
      preconditioner = PreconditionerNone;
    } else if (lower == CG_PRECONDITIONER_JACOBI) {
      preconditioner = PreconditionerJacobi;
    } else {
      std::cerr << "Invalid value for " << CG_PRECONDITIONER << "! ("
                << CG_PRECONDITIONER_NONE << ", or " << CG_PRECONDITIONER_JACOBI
                << ")" << std::endl;
      std::exit(1);
    }

    if (preconditioner != PreconditionerNone &&
        !supportsPreconditioner(preconditioner)) {
      std::cerr << "No support for this preconditioner!" << std::endl;
      std::exit(1);
    }
  }

  env = std::getenv(CG_WORK_DISTRIBUTION);
  if (env != NULL && *env != 0) {
    std::string lower(env);
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

    if (lower == CG_WORK_DISTRIBUTION_BY_ROW) {
      workDistributionCalc = WorkDistributionByRow;
    } else if (lower == CG_WORK_DISTRIBUTION_BY_NZ) {
      workDistributionCalc = WorkDistributionByNz;
    } else {
      std::cerr << "Invalid value for " << CG_WORK_DISTRIBUTION << "! ("
                << CG_WORK_DISTRIBUTION_BY_ROW << ", or "
                << CG_WORK_DISTRIBUTION_BY_NZ << ")" << std::endl;
      std::exit(1);
    }
  }

  env = std::getenv(CG_OVERLAPPED_GATHER);
  if (env != NULL && *env != 0) {
    overlappedGather = (std::string(env) != "0");
    if (overlappedGather &&
        (getNumberOfChunks() == -1 || !supportsOverlappedGather())) {
      std::cerr << "No support for overlapped gather!" << std::endl;
      std::exit(1);
    }
  }
}

// -----------------------------------------------------------------------------
// The functions with allocations cannot live in the header file:
// Otherwise, they are included from openacc/ which makes the PGI compiler use
// page-locked memory for the matrix. That would decrease overall performance.

void CG::allocateMatrixCRS() { matrixCRS.reset(new MatrixCRS); }
void CG::allocateMatrixELL() { matrixELL.reset(new MatrixELL); }
void CG::allocateSplitMatrixCRS() { splitMatrixCRS.reset(new SplitMatrixCRS); }
void CG::allocateSplitMatrixELL() { splitMatrixELL.reset(new SplitMatrixELL); }
void CG::allocatePartitionedMatrixCRS() {
  partitionedMatrixCRS.reset(new PartitionedMatrixCRS);
}
void CG::allocatePartitionedMatrixELL() {
  partitionedMatrixELL.reset(new PartitionedMatrixELL);
}
void CG::allocateJacobi() { jacobi.reset(new Jacobi); }
void CG::allocateK() { k = new floatType[N]; }
void CG::deallocateK() { delete[] k; }
void CG::allocateX() { x = new floatType[N]; }
void CG::deallocateX() { delete[] x; }

// -----------------------------------------------------------------------------

void CG::init(const char *matrixFile) {
  std::cout << "Reading matrix from " << matrixFile << "..." << std::endl;
  auto startIO = now();
  matrixCOO.reset(new MatrixCOO(matrixFile));
  // Copy over size of read matrix.
  N = matrixCOO->N;
  nz = matrixCOO->nz;

  // We count everything from now on as converting!
  auto startConverting = now();

  // Does this implementation need a work distribution?
  int numberOfChunks = getNumberOfChunks();
  if (numberOfChunks != -1) {
    switch (workDistributionCalc) {
    case WorkDistributionByRow:
      workDistribution.reset(
          WorkDistribution::calculateByRow(N, numberOfChunks));
      break;
    case WorkDistributionByNz:
      workDistribution.reset(
          WorkDistribution::calculateByNz(*matrixCOO, numberOfChunks));
      break;
    }
  }

  // Eventually transform the matrix into requested format.
  switch (matrixFormat) {
  case MatrixFormatCOO:
    // Nothing to be done.
    assert(numberOfChunks == -1);
    break;
  case MatrixFormatCRS:
    if (numberOfChunks == -1) {
      std::cout << "Converting matrix to CRS format..." << std::endl;
      allocateMatrixCRS();
      matrixCRS->convert(*matrixCOO);
    } else if (!overlappedGather) {
      std::cout << "Converting and splitting matrix in CRS format..."
                << std::endl;
      allocateSplitMatrixCRS();
      splitMatrixCRS->convert(*matrixCOO, *workDistribution);
    } else {
      std::cout << "Converting and partitioning matrix in CRS format..."
                << std::endl;
      allocatePartitionedMatrixCRS();
      partitionedMatrixCRS->convert(*matrixCOO, *workDistribution);
    }
    break;
  case MatrixFormatELL:
    if (numberOfChunks == -1) {
      std::cout << "Converting matrix to ELL format..." << std::endl;
      allocateMatrixELL();
      matrixELL->convert(*matrixCOO);
    } else if (!overlappedGather) {
      std::cout << "Converting and splitting matrix in ELL format..."
                << std::endl;
      allocateSplitMatrixELL();
      splitMatrixELL->convert(*matrixCOO, *workDistribution);
    } else {
      std::cout << "Converting and partitioning matrix in ELL format..."
                << std::endl;
      allocatePartitionedMatrixELL();
      partitionedMatrixELL->convert(*matrixCOO, *workDistribution);
    }
    break;
  }

  switch (preconditioner) {
  case PreconditionerNone:
    // Nothing to be done.
    break;
  case PreconditionerJacobi:
    std::cout << "Initializing Jacobi preconditioner..." << std::endl;
    allocateJacobi();
    jacobi->init(*matrixCOO);
    break;
  }

  timing.converting = now() - startConverting;
  timing.io = now() - startIO;

  allocateK();
  // Init k so that the solution is (1, ..., 1)^T
  std::memset(k, 0, sizeof(floatType) * N);
  for (int i = 0; i < nz; i++) {
    k[matrixCOO->I[i]] += matrixCOO->V[i];
  }

  allocateX();
  resetX();

  if (matrixFormat != MatrixFormatCOO) {
    // Release matrixCOO which is not needed anymore.
    matrixCOO.reset();
  }
}

void CG::resetX() {
  // Start with (0, ..., 0)^T
  std::memset(x, 0, sizeof(floatType) * N);
}

// #define DEBUG_SOLVE
/// Based on "Methods of Conjugate Gradients for Solving Linear Systems"
/// (http://nvlpubs.nist.gov/nistpubs/jres/049/jresv49n6p409_A1b.pdf)
///
/// Efficient preconditioning is partly based on the following two documents:
///  - http://journals.sagepub.com/doi/pdf/10.1177/109434208700100106
///  - http://www.netlib.org/templates/templates.pdf
void CG::solve() {
  std::cout << "Solving..." << std::endl;
  time_point start = now();

  floatType rho, rho_old;
  floatType r2, nrm2_0;
  floatType dot_pq;
  floatType a, b;

  // r(0) = k - Ax(0) (part of (3:1a))
  matvec(VectorX, VectorR);
  xpay(VectorK, -1.0, VectorR);

  if (preconditioner == PreconditionerNone) {
    // p(0) = r(0) (part of (3:1a))
    cpy(VectorP, VectorR);
  } else {
    // p(0) = B * r(0) (10:4)
    applyPreconditioner(VectorR, VectorP);
  }

  r2 = vectorDot(VectorR, VectorR);
#ifdef DEBUG_SOLVE
  std::cout << "r2 = " << r2 << std::endl;
#endif

  nrm2_0 = std::sqrt(r2);

  if (preconditioner == PreconditionerNone) {
    // rho(0) = |r(0)|^2 (for (3:1b) and (3:1e))
    rho = r2;
  } else {
    // rho(0) = <p(0), r(0)> (for (3:1b) and (3:1e), modified with (10:4))
    rho = vectorDot(VectorP, VectorR);
  }
#ifdef DEBUG_SOLVE
  std::cout << "rho = " << rho << std::endl;
#endif

  for (iteration = 0; iteration < maxIterations; iteration++) {
    // q(i) = A * p(i) (for (3:1b) and (3:1d))
    matvec(VectorP, VectorQ);

    // dot_pq = <p(i), q(i)> (for (3:1b))
    dot_pq = vectorDot(VectorP, VectorQ);
#ifdef DEBUG_SOLVE
    std::cout << "dot_pq = " << dot_pq << std::endl;
#endif

    // a(i) = rho(i) / dot_pq (3:1b)
    a = rho / dot_pq;
#ifdef DEBUG_SOLVE
    std::cout << "a = " << a << std::endl;
#endif

    // x(i + 1) = x(i) + a * p(i) (3:1c)
    axpy(a, VectorP, VectorX);
    // r(i + 1) = r(i) - a * q(i) (3:1d)
    axpy(-a, VectorQ, VectorR);

    r2 = vectorDot(VectorR, VectorR);
#ifdef DEBUG_SOLVE
    std::cout << "r2 = " << r2 << std::endl;
#endif

    // Check convergence with relative residual.
    residual = std::sqrt(r2) / nrm2_0;
    if (residual <= tolerance) {
      // We have (at least partly) done this iteration...
      iteration++;
      break;
    }

    rho_old = rho;
    if (preconditioner == PreconditionerNone) {
      // rho(i + 1) = <r(i + 1), r(i + 1)> (for (3:1b) and (3:1e))
      rho = r2;
    } else {
      // z(i + 1) = B * r(i + 1)
      applyPreconditioner(VectorR, VectorZ);

      // rho(i + 1) = <r(i + 1), z(i + 1)> ((10:4); for (3:1b) and (3:1e))
      rho = vectorDot(VectorR, VectorZ);
    }

    // b(i) = rho(i + 1) / rho(i) (3:1e)
    b = rho / rho_old;
#ifdef DEBUG_SOLVE
    std::cout << "b = " << b << std::endl;
#endif

    if (preconditioner == PreconditionerNone) {
      // p(i + 1) = r(i + 1) + b(i) * p(i) (3:1f)
      xpay(VectorR, b, VectorP);
    } else {
      // p(i + 1) = z(i + 1) + b(i) * p(i)
      xpay(VectorZ, b, VectorP);
    }
  }

  timing.solve = now() - start;
}

bool CG::check() {
  std::cout << "Checking solution..." << std::endl;
  time_point start = now();

  int errors = 0;
  for (int i = 0; i < N; i++) {
    // All elements of x should be 1, see initialization of k.
    if (std::abs(x[i] - 1.0) > checkTolerance) {
      errors++;
    }
  }

  timing.check = now() - start;

  if (errors == 0) {
    std::cout << "Solution is correct!" << std::endl;
  } else {
    std::cout << errors << " elements are incorrect!" << std::endl;
  }

  return (errors == 0);
}

const int maxLabelWidth = 25;
void CG::printPadded(const char *label, const std::string &value) {
  std::cout << std::left << std::setw(maxLabelWidth) << label;
  std::cout << value << std::endl;
}

const int printX = 10;
void CG::printSummary() {
  std::cout << std::endl << "x = [ ";
  for (int i = 0; i < printX && i < N; i++) {
    std::cout << x[i] << " ";
  }
  std::cout << "]" << std::endl;

  printPadded("Iterations:", std::to_string(iteration));
  std::ostringstream oss;
  oss << std::scientific << residual;
  printPadded("Residual:", oss.str());
  printPadded("# rows / # nonzeros:",
              std::to_string(N) + " / " + std::to_string(nz));

  std::string matrixFormatName;
  switch (matrixFormat) {
  case MatrixFormatCOO:
    matrixFormatName = "COO";
    break;
  case MatrixFormatCRS:
    matrixFormatName = "CRS";
    break;
  case MatrixFormatELL:
    matrixFormatName = "ELL";
    break;
  }
  assert(matrixFormatName.length() > 0);
  printPadded("Matrix format:", matrixFormatName);

  std::string preconditionerName;
  switch (preconditioner) {
  case PreconditionerNone:
    preconditionerName = "None";
    break;
  case PreconditionerJacobi:
    preconditionerName = "Jacobi";
    break;
  }
  assert(preconditionerName.length() > 0);
  printPadded("Preconditioner:", preconditionerName);
  if (workDistribution.get() != nullptr) {
    std::string workDistributionName;
    switch (workDistributionCalc) {
    case WorkDistributionByRow:
      workDistributionName = "by row";
      break;
    case WorkDistributionByNz:
      workDistributionName = "by nonzeros";
      break;
    }
    assert(workDistributionName.length() > 0);
    printPadded("Work distribution:", workDistributionName);
    printPadded("Number of chunks:",
                std::to_string(workDistribution->numberOfChunks));

    if (overlappedGather) {
      std::cout << "Overlapped gather with computation!" << std::endl;
    }
  }

  std::cout << std::endl;
  printPadded("IO time:", std::to_string(timing.io.count()));
  double total = 0;
  if (matrixFormat != MatrixFormatCOO || preconditioner != PreconditionerNone) {
    double converting = timing.converting.count();
    printPadded("Converting time:", std::to_string(timing.converting.count()));
    total += converting;
  }

  if (needsTransfer()) {
    double transferTo = timing.transferTo.count();
    printPadded("Transfer to time:", std::to_string(timing.transferTo.count()));
    total += transferTo;
  }
  double solve = timing.solve.count();
  printPadded("Solve time:", std::to_string(solve));
  total += solve;
  if (needsTransfer()) {
    double transferFrom = timing.transferFrom.count();
    printPadded("Transfer from time:", std::to_string(transferFrom));
    total += transferFrom;
  }
  double check = timing.check.count();
  printPadded("Check time:", std::to_string(check));
  total += check;

  printPadded("Total time (excl. IO):", std::to_string(total));

  std::cout << std::endl;
  double matvecTime = timing.matvec.count();
  printPadded("MatVec time:", std::to_string(matvecTime));

  // Don't forget first multiplication!
  const double flops = 2.0 * (iteration + 1) * nz;
  printPadded("MatVec GFLOP/s:", std::to_string(flops / 1e9 / matvecTime));

  printPadded("axpy time:", std::to_string(timing.axpy.count()));
  printPadded("xpay time:", std::to_string(timing.xpay.count()));
  printPadded("vectorDot time:", std::to_string(timing.vectorDot.count()));
  if (preconditioner != PreconditionerNone) {
    printPadded("Preconditioner time:",
                std::to_string(timing.preconditioner.count()));
  }
}

void CG::cleanup() {
  // Uses virtual methods and therefore cannot be done in destructor.
  deallocateK();
  deallocateX();

  if (matrixCRS) {
    matrixCRS->deallocate();
  }
  if (matrixELL) {
    matrixELL->deallocate();
  }

  if (jacobi) {
    jacobi->deallocateC();
  }
}

int main(int argc, char *argv[]) {
  if (argc != 2 && argc != 3) {
    std::cerr << "Usage: " << argv[0] << " <matrix.mtx> [<repetitions>]"
              << std::endl;
    std::exit(1);
  }

  int repetitions = 1;
  if (argc > 2) {
    char *endptr;
    errno = 0;
    int temp = strtol(argv[2], &endptr, 0);
    if (errno == 0 && *endptr == 0 && temp > 0) {
      repetitions = temp;
    } else {
      std::cerr << "Invalid value for repetitions! Disabling..." << std::endl;
    }
  }

#ifdef __PGI
  // The PGI compiler doesn't like freeing this object together with pinned
  // memoy. So leak the memory on purpose...
  CG *cg = CG::getInstance();
#else
  std::unique_ptr<CG> cg(CG::getInstance());
#endif
  cg->parseEnvironment();
  cg->init(argv[1]);

  for (int i = 0; i < repetitions; i++) {
    if (i != 0) {
      std::cout << std::endl;
    }

    if (cg->needsTransfer()) {
      cg->transferTo();
    }
    cg->solve();
    if (cg->needsTransfer()) {
      cg->transferFrom();
    }
    cg->check();
    cg->printSummary();

    // Reset solution vector for next repetition.
    cg->resetX();
    cg->resetNonInitTimings();
  }

  cg->cleanup();

  return EXIT_SUCCESS;
}
