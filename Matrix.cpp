#include <algorithm>
#include <cassert>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>

#include "Matrix.h"
#include "WorkDistribution.h"

MatrixCOO::MatrixCOO(const char *file) {
  try {
    std::ifstream is(file);
    if (!is.is_open()) {
      std::cerr << "Can't open file with matrix!" << std::endl;
      std::exit(1);
    }

    // Read first line.
    std::string line;
    std::stringstream ss;
    getline(is, line);
    ss.str(line);

    std::string banner, mtx, crd, type, storage;
    ss >> banner >> mtx >> crd >> type >> storage;
    // Transform to lower case for comparison.
    std::transform(mtx.begin(), mtx.end(), mtx.begin(), ::tolower);
    std::transform(crd.begin(), crd.end(), crd.begin(), ::tolower);
    std::transform(type.begin(), type.end(), type.begin(), ::tolower);
    std::transform(storage.begin(), storage.end(), storage.begin(), ::tolower);

    if (banner != "\%\%MatrixMarket" || mtx != "matrix" ||
        crd != "coordinate" || type != "real") {
      std::cerr << "Only supporting real matrices in coordinate format!"
                << std::endl;
      std::exit(1);
    }

    bool symmetric = false;
    if (storage == "symmetric") {
      symmetric = true;
    } else if (storage != "general") {
      std::cerr << "Only supporting general or symmetric matrices!"
                << std::endl;
      std::exit(1);
    }

    // Skip following lines with comments.
    do {
      getline(is, line);
    } while (line.size() == 0 || line[0] == '%');

    // Read dimensions.
    int M;
    ss.str(line);
    ss.clear();
    ss >> M >> N >> nz;

    if (N != M) {
      std::cerr << "Need a square matrix!" << std::endl;
      std::exit(1);
    }

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
    std::memset(nzPerRow.get(), 0, sizeof(int) * N);

    // Read matrix.
    for (int i = 0; i < nz; i++) {
      getline(is, line);
      ss.str(line);
      ss.clear();
      ss >> I[i] >> J[i] >> V[i];

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
        V[i] = V[i - 1];

        // Count nz for each row. I[i] is now J[i - 1]!
        nzPerRow[I[i]]++;
      }
    }

    is.close();
  } catch (...) {
    std::cerr << "An exception occurred while reading the matrix!" << std::endl;
    std::exit(1);
  }
}

int MatrixCOO::getMaxNz(int from, int to) const {
  int maxNz = 0;
  for (int i = from; i < to; i++) {
    if (nzPerRow[i] > maxNz) {
      maxNz = nzPerRow[i];
    }
  }
  return maxNz;
}

void MatrixCOO::countNz(const WorkDistribution &wd,
                        std::unique_ptr<int[]> &nzDiag,
                        std::unique_ptr<int[]> &nzMinor) const {
  // Allocate temporary memory.
  nzDiag.reset(new int[N]);
  nzMinor.reset(new int[N]);

  for (int i = 0; i < nz; i++) {
    int row = I[i];
    int chunk = wd.findChunk(row);

    if (wd.isOnDiagonal(chunk, J[i])) {
      nzDiag[row]++;
    } else {
      nzMinor[row]++;
    }
  }
}

template <> DataMatrix<MatrixDataCRS>::DataMatrix(const MatrixCOO &coo) {
  N = coo.N;
  nz = coo.nz;

  // Temporary memory to store current offset in index / value per row.
  std::unique_ptr<int[]> offsets(new int[N]);

  // Construct ptr and initial values for offsets.
  allocatePtr(N);
  ptr[0] = 0;
  for (int i = 1; i <= N; i++) {
    // Copy ptr[i - 1] as initial value for offsets[i - 1].
    offsets[i - 1] = ptr[i - 1];

    ptr[i] = ptr[i - 1] + coo.nzPerRow[i - 1];
  }

  // Construct index and value.
  allocateIndexAndValue(nz);
  for (int i = 0; i < nz; i++) {
    int row = coo.I[i];
    index[offsets[row]] = coo.J[i];
    value[offsets[row]] = coo.V[i];
    offsets[row]++;
  }
}

template <>
SplitMatrix<MatrixDataCRS>::SplitMatrix(const MatrixCOO &coo,
                                        const WorkDistribution &wd) {
  N = coo.N;
  nz = coo.nz;
  allocateData(wd.numberOfChunks);

  // Temporary memory to store current offset in index / value per row.
  std::unique_ptr<int[]> offsets(new int[N]);

  // Construct ptr and initial values for offsets for each chunk.
  for (int c = 0; c < wd.numberOfChunks; c++) {
    int offset = wd.offsets[c];
    int length = wd.lengths[c];

    data[c].allocatePtr(length);
    data[c].ptr[0] = 0;
    for (int i = 1; i <= length; i++) {
      offsets[offset + i - 1] = data[c].ptr[i - 1];

      data[c].ptr[i] = data[c].ptr[i - 1] + coo.nzPerRow[offset + i - 1];
    }
  }

  // Allocate index and value for each chunk.
  for (int c = 0; c < wd.numberOfChunks; c++) {
    int length = wd.lengths[c];
    int values = data[c].ptr[length];
    data[c].allocateIndexAndValue(values);
  }

  // Construct index and value for all chunks.
  for (int i = 0; i < nz; i++) {
    int row = coo.I[i];
    int chunk = wd.findChunk(row);

    data[chunk].index[offsets[row]] = coo.J[i];
    data[chunk].value[offsets[row]] = coo.V[i];
    offsets[row]++;
  }
}

template <>
PartitionedMatrix<MatrixDataCRS>::PartitionedMatrix(
    const MatrixCOO &coo, const WorkDistribution &wd) {
  N = coo.N;
  nz = coo.nz;
  allocateDiagAndMinor(wd.numberOfChunks);

  // Temporary memory to count nonzeros per row.
  std::unique_ptr<int[]> nzDiag, nzMinor;
  coo.countNz(wd, nzDiag, nzMinor);

  // Temporary memory to store current offset in index / value per row.
  std::unique_ptr<int[]> offsetsDiag(new int[N]);
  std::unique_ptr<int[]> offsetsMinor(new int[N]);

  // Construct ptr and initial values for offsets for each chunk.
  for (int c = 0; c < wd.numberOfChunks; c++) {
    int offset = wd.offsets[c];
    int length = wd.lengths[c];

    diag[c].allocatePtr(length);
    minor[c].allocatePtr(length);

    diag[c].ptr[0] = 0;
    minor[c].ptr[0] = 0;

    for (int i = 1; i <= length; i++) {
      offsetsDiag[offset + i - 1] = diag[c].ptr[i - 1];
      offsetsMinor[offset + i - 1] = minor[c].ptr[i - 1];

      diag[c].ptr[i] = diag[c].ptr[i - 1] + nzDiag[offset + i - 1];
      minor[c].ptr[i] = minor[c].ptr[i - 1] + nzMinor[offset + i - 1];
    }
  }

  // Allocate index and value for each chunk.
  for (int c = 0; c < wd.numberOfChunks; c++) {
    int length = wd.lengths[c];
    int valuesDiag = diag[c].ptr[length];
    int valuesMinor = diag[c].ptr[length];

    diag[c].allocateIndexAndValue(valuesDiag);
    minor[c].allocateIndexAndValue(valuesMinor);
  }

  // Construct index and value for all chunks.
  for (int i = 0; i < nz; i++) {
    int row = coo.I[i];
    int column = coo.J[i];
    int chunk = wd.findChunk(row);

    if (wd.isOnDiagonal(chunk, column)) {
      diag[chunk].index[offsetsDiag[row]] = column;
      diag[chunk].value[offsetsDiag[row]] = coo.V[i];
      offsetsDiag[row]++;
    } else {
      minor[chunk].index[offsetsMinor[row]] = column;
      minor[chunk].value[offsetsMinor[row]] = coo.V[i];
      offsetsMinor[row]++;
    }
  }
}

template <> DataMatrix<MatrixDataELL>::DataMatrix(const MatrixCOO &coo) {
  N = coo.N;
  nz = coo.nz;
  elements = N * coo.getMaxNz();

  // Copy over already collected nonzeros per row.
  allocateLength(N);
  std::memcpy(length, coo.nzPerRow.get(), sizeof(int) * N);

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

template <>
SplitMatrix<MatrixDataELL>::SplitMatrix(const MatrixCOO &coo,
                                        const WorkDistribution &wd) {
  N = coo.N;
  nz = coo.nz;
  allocateData(wd.numberOfChunks);

  // Allocate length for each chunk.
  for (int c = 0; c < wd.numberOfChunks; c++) {
    int offset = wd.offsets[c];
    int length = wd.lengths[c];
    int maxNz = coo.getMaxNz(offset, offset + length);

    // Copy over already collected nonzeros per row.
    data[c].allocateLength(length);
    std::memcpy(data[c].length, coo.nzPerRow.get() + offset,
                sizeof(int) * length);

    data[c].elements = maxNz * length;
    data[c].allocateIndexAndData();
  }

  // Temporary memory to store current offset in index / value per row.
  std::unique_ptr<int[]> offsets(new int[N]);
  std::memset(offsets.get(), 0, sizeof(int) * N);

  // Construct column and data for all chunks.
  for (int i = 0; i < nz; i++) {
    int row = coo.I[i];
    int chunk = wd.findChunk(row);

    int k = offsets[row] * wd.lengths[chunk] + row - wd.offsets[chunk];
    data[chunk].index[k] = coo.J[i];
    data[chunk].data[k] = coo.V[i];
    offsets[row]++;
  }
}

template <>
PartitionedMatrix<MatrixDataELL>::PartitionedMatrix(
    const MatrixCOO &coo, const WorkDistribution &wd) {
  N = coo.N;
  nz = coo.nz;
  allocateDiagAndMinor(wd.numberOfChunks);

  // Temporary memory to count nonzeros per row.
  std::unique_ptr<int[]> nzDiag, nzMinor;
  coo.countNz(wd, nzDiag, nzMinor);

  // Allocate length for each chunk.
  for (int c = 0; c < wd.numberOfChunks; c++) {
    int offset = wd.offsets[c];
    int length = wd.lengths[c];
    int maxNzDiag = 0, maxNzMinor = 0;
    for (int i = offset; i < offset + length; i++) {
      if (nzDiag[i] > maxNzDiag) {
        maxNzDiag = nzDiag[i];
      }
      if (nzMinor[i] > maxNzMinor) {
        maxNzMinor = nzMinor[i];
      }
    }

    // Copy over already collected nonzeros per row.
    diag[c].allocateLength(length);
    std::memcpy(diag[c].length, nzDiag.get() + offset, sizeof(int) * length);
    minor[c].allocateLength(length);
    std::memcpy(minor[c].length, nzMinor.get() + offset, sizeof(int) * length);

    diag[c].elements = maxNzDiag * length;
    diag[c].allocateIndexAndData();
    minor[c].elements = maxNzMinor * length;
    minor[c].allocateIndexAndData();
  }

  // Temporary memory to store current offset in index / value per row.
  std::unique_ptr<int[]> offsetsDiag(new int[N]);
  std::memset(offsetsDiag.get(), 0, sizeof(int) * N);
  std::unique_ptr<int[]> offsetsMinor(new int[N]);
  std::memset(offsetsMinor.get(), 0, sizeof(int) * N);

  // Construct column and data for all chunks.
  for (int i = 0; i < nz; i++) {
    int row = coo.I[i];
    int column = coo.J[i];
    int chunk = wd.findChunk(row);

    if (wd.isOnDiagonal(chunk, column)) {
      int k = offsetsDiag[row] * wd.lengths[chunk] + row - wd.offsets[chunk];
      diag[chunk].index[k] = coo.J[i];
      diag[chunk].data[k] = coo.V[i];
      offsetsDiag[row]++;
    } else {
      int k = offsetsMinor[row] * wd.lengths[chunk] + row - wd.offsets[chunk];
      minor[chunk].index[k] = coo.J[i];
      minor[chunk].data[k] = coo.V[i];
      offsetsMinor[row]++;
    }
  }
}
