/*
    Copyright (C) 2017  Jonas Hahnfeld

    This file is part of CGxx.

    CGxx is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    CGxx is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with CGxx.  If not, see <http://www.gnu.org/licenses/>. */

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
  void init(const MatrixCOO &coo);

  /// Allocate #C.
  virtual void allocateC(int N);
  /// Deallocate #C.
  virtual void deallocateC();
};

#endif
