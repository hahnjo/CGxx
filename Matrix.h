#ifndef MATRIX_H
#define MATRIX_H

#include <memory>

#include "def.h"

struct Matrix {
  int N;
  int nz;
};

struct MatrixCOO : Matrix {
  std::unique_ptr<int[]> I;
  std::unique_ptr<int[]> J;
  std::unique_ptr<floatType[]> V;

  std::unique_ptr<int[]> nzPerRow;

  MatrixCOO() = delete;
  MatrixCOO(const char *file);

  int getMaxNz() const;
};

struct MatrixCRS : Matrix {
  std::unique_ptr<int[]> ptr;
  std::unique_ptr<int[]> index;
  std::unique_ptr<floatType[]> value;

  MatrixCRS() = delete;
  MatrixCRS(const MatrixCOO &coo);

  virtual void allocatePtr() { ptr.reset(new int[N + 1]); }
  virtual void allocateIndexAndValue() {
    index.reset(new int[nz]);
    value.reset(new floatType[nz]);
  }
};

struct MatrixELL : Matrix {
  int maxNz;
  int elements;

  std::unique_ptr<int[]> length;
  std::unique_ptr<int[]> index;
  std::unique_ptr<floatType[]> data;

  MatrixELL() = delete;
  MatrixELL(const MatrixCOO &coo);

  virtual void allocateLength() { length.reset(new int[N]); }
  virtual void allocateIndexAndData() {
    index.reset(new int[elements]);
    data.reset(new floatType[elements]);
  }
};

#endif
