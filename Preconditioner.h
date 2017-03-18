#ifndef PRECONDITIONER_H
#define PRECONDITIONER_H

#include <memory>

#include "Matrix.h"
#include "def.h"

/// %Jacobi preconditioner.
struct Jacobi {
  /// Reciprocal diagonal elements of the matrix.
  std::unique_ptr<floatType[]> C;

  /// Initialize object with \a coo for an efficient %Jacobi preconditioner.
  Jacobi(const MatrixCOO &coo);

  /// Allocate #C.
  virtual void allocateC(int N) { C.reset(new floatType[N]); }
};

#endif
