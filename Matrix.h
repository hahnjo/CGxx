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
  virtual void allocateData() { data.reset(new Data[numberOfChunks]); }

  ~SplitMatrix() {
    if (data) {
      for (int i = 0; i < numberOfChunks; i++) {
        data[i].deallocate();
      }
    }
  }
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
  virtual void allocateDiagAndMinor() {
    diag.reset(new Data[numberOfChunks]);
    minor.reset(new Data[numberOfChunks]);
  }

  ~PartitionedMatrix() {
    if (diag) {
      for (int i = 0; i < numberOfChunks; i++) {
        diag[i].deallocate();
      }
    }

    if (minor) {
      for (int i = 0; i < numberOfChunks; i++) {
        minor[i].deallocate();
      }
    }
  }
};
using PartitionedMatrixCRS = PartitionedMatrix<MatrixDataCRS>;
using PartitionedMatrixELL = PartitionedMatrix<MatrixDataELL>;

#endif
