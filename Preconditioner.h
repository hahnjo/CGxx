#ifndef PRECONDITIONER_H
#define PRECONDITIONER_H

#include <memory>

#include "Matrix.h"
#include "def.h"

/// %Jacobi preconditioner.
struct Jacobi {
  /// Reciprocal diagonal elements of the matrix.
  floatType *C;

  /// Initialize object with \a coo for an efficient %Jacobi preconditioner.
  Jacobi(const MatrixCOO &coo);

  ~Jacobi() { deallocateC(); }

  /// Allocate #C.
  virtual void allocateC(int N) { C = new floatType[N]; }
  /// Deallocate #C.
  virtual void deallocateC() { delete[] C; }
};

#endif
