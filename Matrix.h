#ifndef MATRIX_H
#define MATRIX_H

#include <memory>

#include "WorkDistribution.h"
#include "def.h"

/// Base class for storing a sparse matrix.
struct Matrix {
  /// Dimension of this matrix.
  int N;
  /// Nonzeros in this matrix.
  int nz;
};

/// %Matrix stored in coordinate format.
struct MatrixCOO : Matrix {
  /// Array of rows.
  std::unique_ptr<int[]> I;
  /// Array of columns.
  std::unique_ptr<int[]> J;
  /// Values in the matrix.
  std::unique_ptr<floatType[]> V;

  /// Nonzeros in a row.
  std::unique_ptr<int[]> nzPerRow;

  MatrixCOO() = delete;
  /// Read matrix in coordinate from \a file.
  MatrixCOO(const char *file);

  /// Get maximum number of nonzeros in a row.
  int getMaxNz() const { return getMaxNz(0, N); }
  /// Get maximum number of nonzeros in a row between \a from and \a to.
  int getMaxNz(int from, int to) const;
};

/// Data for storing a matrix in CRS format.
struct MatrixCRSData {
  /// Start index in #index and #value for a given row.
  int *ptr;
  /// Array of column indices.
  int *index;
  /// Values in the matrix.
  floatType *value;

  ~MatrixCRSData() {
    deallocatePtr();
    deallocateIndexAndValue();
  }

  /// Allocate #ptr.
  virtual void allocatePtr(int rows) { ptr = new int[rows + 1]; }
  /// Deallocate #ptr.
  virtual void deallocatePtr() { delete[] ptr; }
  /// Allocate #index and #value.
  virtual void allocateIndexAndValue(int values) {
    index = new int[values];
    value = new floatType[values];
  }
  /// Deallocate #index and #value.
  virtual void deallocateIndexAndValue() {
    delete[] index;
    delete[] value;
  }
};

/// %Matrix stored in CRS format.
struct MatrixCRS : Matrix, MatrixCRSData {
  MatrixCRS() = delete;
  /// Convert \a coo into CRS format.
  MatrixCRS(const MatrixCOO &coo);
};

/// %Matrix stored in CRS format, split for a WorkDistribution.
struct SplitMatrixCRS : Matrix {
  /// Data for each chunk of the WorkDistribution.
  std::unique_ptr<MatrixCRSData[]> data;

  SplitMatrixCRS() = delete;
  /// Convert \a coo into CRS format and split based on \a wd.
  SplitMatrixCRS(const MatrixCOO &coo, const WorkDistribution &wd);

  virtual void allocateData(int numberOfChunks) {
    data.reset(new MatrixCRSData[numberOfChunks]);
  }
};

/// Data for storing a matrix in ELLPACK format.
struct MatrixELLData {
  /// Maximum number of nonzeros in a row.
  /// @see MatrixCOO#getMaxNz()
  int maxNz;
  /// Elements in #index and #data including padding.
  int elements;

  /// Array of length, holding number of nonzeros per row.
  int *length;
  /// Array of column indices.
  int *index;
  /// Data in the matrix.
  floatType *data;

  ~MatrixELLData() {
    deallocateLength();
    deallocateIndexAndData();
  }

  /// Allocate #length.
  virtual void allocateLength(int rows) { length = new int[rows]; }
  /// Deallocate #length.
  virtual void deallocateLength() { delete[] length; }
  /// Allocate #index and #data.
  virtual void allocateIndexAndData() {
    index = new int[elements];
    data = new floatType[elements];
  }
  /// Deallocate #index and #data.
  virtual void deallocateIndexAndData() {
    delete[] index;
    delete[] data;
  }
};

/// %Matrix stored in ELLPACK format.
struct MatrixELL : Matrix, MatrixELLData {
  MatrixELL() = delete;
  /// Convert \a coo into ELLPACK format.
  MatrixELL(const MatrixCOO &coo);
};

/// %Matrix stored in ELLPACK format, split for a WorkDistribution.
struct SplitMatrixELL : Matrix {
  /// Data for each chunk of the WorkDistribution.
  std::unique_ptr<MatrixELLData[]> data;

  SplitMatrixELL() = delete;
  /// Convert \a coo into ELLPACK format and split based on \a wd.
  SplitMatrixELL(const MatrixCOO &coo, const WorkDistribution &wd);

  virtual void allocateData(int numberOfChunks) {
    data.reset(new MatrixELLData[numberOfChunks]);
  }
};

#endif
