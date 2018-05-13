// SPDX-License-Identifier:	GPL-3.0-or-later

#include "Preconditioner.h"
#include "Matrix.h"

// The functions for allocation and deallocation cannot live in the header file:
// Otherwise, they are included from openacc/ which makes the PGI compiler use
// page-locked memory for the preconditioner. That would decrease performance.

void Jacobi::allocateC(int N) { C = new floatType[N]; }
void Jacobi::deallocateC() { delete[] C; }

void Jacobi::init(const MatrixCOO &coo) {
  allocateC(coo.N);

  for (int i = 0; i < coo.nz; i++) {
    if (coo.I[i] != coo.J[i]) {
      // We need to find the diagonal elements.
      continue;
    }
    C[coo.I[i]] = 1 / coo.V[i];
  }
}
