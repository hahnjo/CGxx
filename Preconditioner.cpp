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
