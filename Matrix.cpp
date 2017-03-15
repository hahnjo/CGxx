#include <cstdio>
#include <cstring>
#include <iostream>

#include "Matrix.h"

extern "C" {
#include "mmio.h"
};

MatrixCOO::MatrixCOO(const char *file) {
  FILE *fp = fopen(file, "r");
  if (fp == NULL) {
    std::cerr << "ERROR: Can't open file!" << std::endl;
    std::exit(1);
  }

  MM_typecode matcode;
  if (mm_read_banner(fp, &matcode) != 0) {
    std::cerr << "ERROR: Could not process Matrix Market banner!" << std::endl;
    std::exit(1);
  }

  // Check properties.
  if (!mm_is_sparse(matcode) || !mm_is_real(matcode)) {
    std::cerr << "ERROR: Only supporting real matrixes in coordinate format!";
    std::cerr << " (type: " << mm_typecode_to_str(matcode) << ")" << std::endl;
    std::exit(1);
  }

  int M;
  if (mm_read_mtx_crd_size(fp, &M, &N, &nz) != 0) {
    std::cerr << "ERROR: Could not read matrix size!" << std::endl;
    std::exit(1);
  }

  if (N != M) {
    std::cerr << "ERROR: Need a quadratic matrix!" << std::endl;
    std::exit(1);
  }

  bool symmetric = mm_is_symmetric(matcode);
  if (symmetric) {
    // Store upper and lower triangular!
    nz = 2 * nz - N;
  }

  // Allocate memory. No implementation will need to "optimize" this because
  // MatrixCOO is not really meant to be used in "real" computations.
  I.reset(new int[nz]);
  J.reset(new int[nz]);
  V.reset(new floatType[nz]);
  nzPerRow.reset(new int[N]);

  // Read matrix.
  for (int i = 0; i < nz; i++) {
    fscanf(fp, "%d %d %lg\n", &I[i], &J[i], &V[i]);

    // Adjust from 1-based to 0-based.
    I[i]--;
    J[i]--;

    // Count nz for each row.
    nzPerRow[I[i]]++;

    // If not on the main diagonal, we have to duplicate the entry.
    if (symmetric && I[i] != J[i]) {
      i++;
      I[i] = J[i - 1];
      J[i] = I[i - 1];

      // Count nz for each row. I[i] is now J[i -1]!
      nzPerRow[I[i]]++;
    }
  }

  fclose(fp);
}

int MatrixCOO::getMaxNz() const {
  int maxNz = 0;
  for (int i = 0; i < N; i++) {
    if (nzPerRow[i] > maxNz) {
      maxNz = nzPerRow[i];
    }
  }
  return maxNz;
}

MatrixCRS::MatrixCRS(const MatrixCOO &coo) {
  N = coo.N;
  nz = coo.nz;

  // Temporary memory to store current offset in index / value per row.
  std::unique_ptr<int[]> offsets(new int[N]);

  // Construct ptr and initial values for offsets.
  allocatePtr();
  ptr[0] = 0;
  for (int i = 1; i <= N; i++) {
    // Copy ptr[i - 1] as initial value for offsets[i - 1].
    offsets[i - 1] = ptr[i - 1];

    ptr[i] = ptr[i - 1] + coo.nzPerRow[i - 1];
  }

  // Construct index and value.
  allocateIndexAndValue();
  for (int i = 0; i < nz; i++) {
    int row = coo.I[i];
    index[offsets[row]] = coo.J[i];
    value[offsets[row]] = coo.V[i];
    offsets[row]++;
  }
}

MatrixELL::MatrixELL(const MatrixCOO &coo) {
  N = coo.N;
  nz = coo.nz;
  elements = N * coo.getMaxNz();

  // Copy over already collected nonzeros per row.
  allocateLength();
  std::memcpy(length.get(), coo.nzPerRow.get(), sizeof(int) * N);

  // Temporary memory to store current offset in index / value per row.
  std::unique_ptr<int[]> offsets(new int[N]);
  std::memset(offsets.get(), 0, sizeof(int) * N);

  // Construct column and data.
  allocateIndexAndData();
  for (int i = 0; i < nz; i++) {
    int row = coo.I[i];
    int k = offsets[row] * N + row;
    index[k] = coo.J[i];
    data[k] = coo.V[i];
    offsets[row]++;
  }
}
