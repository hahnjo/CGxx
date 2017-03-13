#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>

#include "CG.h"
#include "Matrix.h"

const char *CG_MAX_ITER = "CG_MAX_ITER";
const char *CG_TOLERANCE = "CG_TOLERANCE";

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
}

void CG::allocateK() { k.reset(new floatType[N]); }

void CG::allocateX() { x.reset(new floatType[N]); }

void CG::init(const char *matrixFile) {
  matrixCOO.reset(new MatrixCOO);
  matrixCOO->readFromFile(matrixFile);
  // Copy over size of read matrix.
  N = matrixCOO->N;
  nz = matrixCOO->nz;

  allocateK();
  // Init k so that the solution is (1, ..., 1)^T
  std::memset(k.get(), 0, sizeof(floatType) * N);
  for (int i = 0; i < nz; i++) {
    k[matrixCOO->I[i]] += matrixCOO->V[i];
  }

  allocateX();
  // Start with (0, ..., 0)^T
  std::memset(x.get(), 0, sizeof(floatType) * N);
}

// #define DEBUG_SOLVE
void CG::solve() {
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

  timing.total = now() - start;
}

const int maxLabelWidth = 20;
void CG::printPadded(const char *label, const std::string &value) {
  std::cout << std::left << std::setw(maxLabelWidth) << label;
  std::cout << value << std::endl;
}

const int printX = 10;
void CG::printSummary() {
  std::cout << "x = [ ";
  for (int i = 0; i < printX && i < N; i++) {
    std::cout << x[i] << " ";
  }
  std::cout << "]" << std::endl;

  printPadded("Iterations:", std::to_string(iteration));
  printPadded("Residual:", std::to_string(residual));

  double matvecTime = timing.matvec.count();
  printPadded("MatVec time:", std::to_string(matvecTime));

  // Don't forget first multiplication!
  const double flops = 2.0 * (iteration + 1) * nz;
  printPadded("MatVec GFLOP/s:", std::to_string(flops / 1e9 / matvecTime));

  printPadded("axpy time:", std::to_string(timing.axpy.count()));
  printPadded("xpay time:", std::to_string(timing.xpay.count()));
  printPadded("vectorDot time:", std::to_string(timing.xpay.count()));
  printPadded("Total time:", std::to_string(timing.total.count()));
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
