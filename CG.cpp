#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <locale>
#include <memory>
#include <sstream>

#include "CG.h"
#include "Matrix.h"

const char *CG_MAX_ITER = "CG_MAX_ITER";
const char *CG_TOLERANCE = "CG_TOLERANCE";

const char *CG_MATRIX_FORMAT = "CG_MATRIX_FORMAT";
const char *CG_MATRIX_FORMAT_COO = "COO";
const char *CG_MATRIX_FORMAT_CRS = "CRS";
const char *CG_MATRIX_FORMAT_ELL = "ELL";

void CG::parseEnvironment() {
  const char *env;
  char *endptr;

  env = std::getenv(CG_MAX_ITER);
  if (env != NULL && *env != 0) {
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
    floatType tolerance = strtod(env, &endptr);
    if (errno == 0 && *endptr == 0 && tolerance > 0) {
      this->tolerance = tolerance;
    } else {
      std::cerr << "Invalid value for " << CG_TOLERANCE << "!" << std::endl;
      std::exit(1);
    }
  }

  env = std::getenv(CG_MATRIX_FORMAT);
  if (env != NULL && *env != 0) {
    std::string upper(env);
    std::transform(upper.begin(), upper.end(), upper.begin(),
                   [](char c) { return std::toupper(c); });

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
}

void CG::init(const char *matrixFile) {
  std::cout << "Reading matrix from " << matrixFile << "..." << std::endl;
  auto startIO = now();
  matrixCOO.reset(new MatrixCOO(matrixFile));
  // Copy over size of read matrix.
  N = matrixCOO->N;
  nz = matrixCOO->nz;

  // Eventually transform the matrix into requested format.
  auto startConverting = now();
  switch (matrixFormat) {
  case MatrixFormatCRS:
    std::cout << "Converting matrix to CRS format..." << std::endl;
    convertToMatrixCRS();
    break;
  case MatrixFormatELL:
    std::cout << "Converting matrix to ELL format..." << std::endl;
    convertToMatrixELL();
    break;
  }
  timing.converting = now() - startConverting;
  timing.io = now() - startIO;

  allocateK();
  // Init k so that the solution is (1, ..., 1)^T
  std::memset(k.get(), 0, sizeof(floatType) * N);
  for (int i = 0; i < nz; i++) {
    k[matrixCOO->I[i]] += matrixCOO->V[i];
  }

  allocateX();
  // Start with (0, ..., 0)^T
  std::memset(x.get(), 0, sizeof(floatType) * N);

  if (matrixFormat != MatrixFormatCOO) {
    // Release matrixCOO which is not needed anymore.
    matrixCOO.reset();
  }
}

// #define DEBUG_SOLVE
/// Based on "Methods of Conjugate Gradients for Solving Linear Systems"
/// (http://nvlpubs.nist.gov/nistpubs/jres/049/jresv49n6p409_A1b.pdf)
void CG::solve() {
  std::cout << "Solving..." << std::endl;
  auto start = now();

  floatType r2, r2_old;
  floatType nrm2_0, nrm2;
  floatType dot_pq;
  floatType a, b;

  // p(0) = r(0) = k - Ax(0) (3:1a)
  matvec(VectorX, VectorR);
  xpay(VectorK, -1.0, VectorR);
  cpy(VectorP, VectorR);

  // r2(0) = |r(0)|^2 (for (3:1b) and (3:1e))
  r2 = vectorDot(VectorR, VectorR);
#ifdef DEBUG_SOLVE
  std::cout << "r2 = " << r2 << std::endl;
#endif

  nrm2 = std::sqrt(r2);

  for (iteration = 0; iteration < maxIterations; iteration++) {
    // q(i) = A * p(i) (for (3:1b) and (3:1d))
    matvec(VectorP, VectorQ);

    // dot_pq = <p(i), q(i)> (for (3:1b))
    dot_pq = vectorDot(VectorP, VectorQ);
#ifdef DEBUG_SOLVE
    std::cout << "dot_pq = " << dot_pq << std::endl;
#endif

    // a(i) = rho(i) / dot_pq (3:1b)
    a = r2 / dot_pq;
#ifdef DEBUG_SOLVE
    std::cout << "a = " << a << std::endl;
#endif

    // x(i + 1) = x(i) + a * p(i) (3:1c)
    axpy(a, VectorP, VectorX);
    // r(i + 1) = r(i) - a * q(i) (3:1d)
    axpy(-a, VectorQ, VectorR);

    r2_old = r2;
    // r2(i + 1) = |r(i + 1)|^2 (for (3:1b) and (3:1e))
    r2 = vectorDot(VectorR, VectorR);
#ifdef DEBUG_SOLVE
    std::cout << "r2 = " << r2 << std::endl;
#endif

    // Check convergence with relative residual.
    residual = std::sqrt(r2) / nrm2;
    if (residual <= tolerance) {
      // We have (at least partly) done this iteration...
      iteration++;
      break;
    }

    // b(i) = |r(i + 1)|^2 / |r(i)|^2 (3:1e)
    b = r2 / r2_old;
#ifdef DEBUG_SOLVE
    std::cout << "b = " << b << std::endl;
#endif

    // p(i + 1) = r(i + 1) + b(i) * p(i) (3:1f)
    xpay(VectorR, b, VectorP);
  }

  timing.solve = now() - start;
}

const int maxLabelWidth = 20;
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

  std::cout << std::endl;
  printPadded("IO time:", std::to_string(timing.io.count()));
  if (matrixFormat != MatrixFormatCOO) {
    printPadded("Converting time:", std::to_string(timing.converting.count()));
  }

  printPadded("Solve time:", std::to_string(timing.solve.count()));
  double matvecTime = timing.matvec.count();
  printPadded("MatVec time:", std::to_string(matvecTime));

  // Don't forget first multiplication!
  const double flops = 2.0 * (iteration + 1) * nz;
  printPadded("MatVec GFLOP/s:", std::to_string(flops / 1e9 / matvecTime));

  printPadded("axpy time:", std::to_string(timing.axpy.count()));
  printPadded("xpay time:", std::to_string(timing.xpay.count()));
  printPadded("vectorDot time:", std::to_string(timing.xpay.count()));
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <matrix.mtx>" << std::endl;
    std::exit(1);
  }

  std::unique_ptr<CG> cg(CG::getInstance());
  cg->parseEnvironment();
  cg->init(argv[1]);

  cg->solve();

  cg->printSummary();

  return EXIT_SUCCESS;
}
