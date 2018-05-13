// SPDX-License-Identifier:	GPL-3.0-or-later

#ifndef MATRIX_H
#define MATRIX_H

#include <memory>

#include "def.h"

// Forward declaration to not include WorkDistribution.h
struct WorkDistribution;

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

  /// @return number of nonzeros for each chunk in \a wd.
  void countNz(const WorkDistribution &wd, std::unique_ptr<int[]> &nzDiag,
               std::unique_ptr<int[]> &nzMinor) const;
};

/// Data for storing a matrix in CRS format.
struct MatrixDataCRS {
  /// Start index in #index and #value for a given row.
  int *ptr;
  /// Array of column indices.
  int *index;
  /// Values in the matrix.
  floatType *value;

  /// Allocate #ptr.
  virtual void allocatePtr(int rows);
  /// Deallocate #ptr.
  virtual void deallocatePtr();
  /// Allocate #index and #value.
  virtual void allocateIndexAndValue(int values);
  /// Deallocate #index and #value.
  virtual void deallocateIndexAndValue();

  void deallocate() {
    deallocatePtr();
    deallocateIndexAndValue();
  }
};

/// Data for storing a matrix in ELLPACK format.
struct MatrixDataELL {
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

  /// Allocate #length.
  virtual void allocateLength(int rows);
  /// Deallocate #length.
  virtual void deallocateLength();
  /// Allocate #index and #data.
  virtual void allocateIndexAndData();
  /// Deallocate #index and #data.
  virtual void deallocateIndexAndData();

  void deallocate() {
    deallocateLength();
    deallocateIndexAndData();
  }
};

/// %Matrix with specified data.
template <class Data> struct DataMatrix : Matrix, Data {
  /// Convert \a coo.
  void convert(const MatrixCOO &coo);
};
using MatrixCRS = DataMatrix<MatrixDataCRS>;
using MatrixELL = DataMatrix<MatrixDataELL>;

/// %Matrix split for a WorkDistribution.
template <class Data> struct SplitMatrix : Matrix {
  /// Number of chunks in this matrix.
  int numberOfChunks;

  /// Data for each chunk of the WorkDistribution.
  std::unique_ptr<Data[]> data;

  /// Convert \a coo and split based on \a wd.
  void convert(const MatrixCOO &coo, const WorkDistribution &wd);

  /// Allocate #data.
  virtual void allocateData();

  ~SplitMatrix();
};
using SplitMatrixCRS = SplitMatrix<MatrixDataCRS>;
using SplitMatrixELL = SplitMatrix<MatrixDataELL>;

/// %Matrix partitioned for a WorkDistribution.
template <class Data> struct PartitionedMatrix : Matrix {
  /// Number of chunks in this matrix.
  int numberOfChunks;

  /// Data on the diagonal for each chunk of the WorkDistribution.
  std::unique_ptr<Data[]> diag;

  /// Data NOT on the diagonal for each chunk of the WorkDistribution.
  std::unique_ptr<Data[]> minor;

  /// Convert \a coo and partition based on \a wd.
  void convert(const MatrixCOO &coo, const WorkDistribution &wd);

  /// Allocate #diag and #minor.
  virtual void allocateDiagAndMinor();

  ~PartitionedMatrix();
};
using PartitionedMatrixCRS = PartitionedMatrix<MatrixDataCRS>;
using PartitionedMatrixELL = PartitionedMatrix<MatrixDataELL>;

#endif
