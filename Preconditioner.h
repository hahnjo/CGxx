// SPDX-License-Identifier:	GPL-3.0-or-later

#ifndef PRECONDITIONER_H
#define PRECONDITIONER_H

#include <memory>

#include "Matrix.h"
#include "def.h"

/// %Jacobi preconditioner.
struct Jacobi {
  /// Reciprocal diagonal elements of the matrix.
  floatType *C;

  virtual ~Jacobi() { }

  /// Initialize object with \a coo for an efficient %Jacobi preconditioner.
  void init(const MatrixCOO &coo);

  /// Allocate #C.
  virtual void allocateC(int N);
  /// Deallocate #C.
  virtual void deallocateC();
};

#endif
