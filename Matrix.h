#ifndef MATRIX_H
#define MATRIX_H

#include <memory>

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

  /// Get maximum number of nonzers in a row.
  int getMaxNz() const;
};

/// %Matrix stored in CRS format.
struct MatrixCRS : Matrix {
  /// Start index in #index and #value for a given row.
  std::unique_ptr<int[]> ptr;
  /// Array of column indices.
  std::unique_ptr<int[]> index;
  /// Values in the matrix.
  std::unique_ptr<floatType[]> value;

  MatrixCRS() = delete;
  /// Convert \a coo into CRS format.
  MatrixCRS(const MatrixCOO &coo);

  /// Allocate #ptr.
  virtual void allocatePtr() { ptr.reset(new int[N + 1]); }
  /// Allocate #index and #value.
  virtual void allocateIndexAndValue() {
    index.reset(new int[nz]);
    value.reset(new floatType[nz]);
  }
};

/// %Matrix stored in ELLPACK format.
struct MatrixELL : Matrix {
  /// Maximum number of nonzeros in a row.
  /// @see MatrixCOO#getMaxNz()
  int maxNz;
  /// Elements in #index and #data including padding.
  int elements;

  /// Array of length, holding number of nonzeros per row.
  std::unique_ptr<int[]> length;
  /// Array of column indices.
  std::unique_ptr<int[]> index;
  /// Data in the matrix.
  std::unique_ptr<floatType[]> data;

  MatrixELL() = delete;
  /// Convert \a coo into ELLPACK format.
  MatrixELL(const MatrixCOO &coo);

  /// Allocate #length.
  virtual void allocateLength() { length.reset(new int[N]); }
  /// Allocate #index and #data.
  virtual void allocateIndexAndData() {
    index.reset(new int[elements]);
    data.reset(new floatType[elements]);
  }
};

#endif
