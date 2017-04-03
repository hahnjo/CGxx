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

  ~MatrixDataCRS() {
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

  ~MatrixDataELL() {
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

/// %Matrix with specified data.
template <class Data> struct DataMatrix : Matrix, Data {
  DataMatrix() = delete;
  /// Convert \a coo.
  DataMatrix(const MatrixCOO &coo);
};
using MatrixCRS = DataMatrix<MatrixDataCRS>;
using MatrixELL = DataMatrix<MatrixDataELL>;

/// %Matrix split for a WorkDistribution.
template <class Data> struct SplitMatrix : Matrix {
  /// Data for each chunk of the WorkDistribution.
  std::unique_ptr<Data[]> data;

  SplitMatrix() = delete;
  /// Convert \a coo and split based on \a wd.
  SplitMatrix(const MatrixCOO &coo, const WorkDistribution &wd);

  /// Allocate #data.
  virtual void allocateData(int numberOfChunks) {
    data.reset(new Data[numberOfChunks]);
  }
};
using SplitMatrixCRS = SplitMatrix<MatrixDataCRS>;
using SplitMatrixELL = SplitMatrix<MatrixDataELL>;

/// %Matrix partitioned for a WorkDistribution.
template <class Data> struct PartitionedMatrix : Matrix {
  /// Data on the diagonal for each chunk of the WorkDistribution.
  std::unique_ptr<Data[]> diag;

  /// Data NOT on the diagonal for each chunk of the WorkDistribution.
  std::unique_ptr<Data[]> minor;

  PartitionedMatrix() = delete;
  /// Convert \a coo and partition based on \a wd.
  PartitionedMatrix(const MatrixCOO &coo, const WorkDistribution &wd);

  /// Allocate #diag and #minor.
  virtual void allocateDiagAndMinor(int numberOfChunks) {
    diag.reset(new Data[numberOfChunks]);
    minor.reset(new Data[numberOfChunks]);
  }
};
using PartitionedMatrixCRS = PartitionedMatrix<MatrixDataCRS>;
using PartitionedMatrixELL = PartitionedMatrix<MatrixDataELL>;

#endif
