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

#include "Preconditioner.h"
#include "Matrix.h"

Jacobi::Jacobi(const MatrixCOO &coo) {
  allocateC(coo.N);

  for (int i = 0; i < coo.nz; i++) {
    if (coo.I[i] != coo.J[i]) {
      // We need to find the diagonal elements.
      continue;
    }
    C[coo.I[i]] = 1 / coo.V[i];
  }
}
