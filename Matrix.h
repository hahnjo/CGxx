#ifndef MATRIX_H
#define MATRIX_H

#include <memory>

#include "def.h"

struct Matrix {
  int N;
  int nz;

  Matrix() = default;
  Matrix(int N, int nz) : N(N), nz(nz) {}
};

struct MatrixCOO : Matrix {
  std::unique_ptr<int[]> I;
  std::unique_ptr<int[]> J;
  std::unique_ptr<floatType[]> V;

  std::unique_ptr<int[]> nzPerRow;

  MatrixCOO() = default;

  void readFromFile(const char *file);
  int getMaxNz() const;
};

struct MatrixCRS : Matrix {
  std::unique_ptr<int[]> ptr;
  std::unique_ptr<int[]> index;
  std::unique_ptr<floatType[]> value;

  MatrixCRS() = delete;
  MatrixCRS(int N, int nz) : Matrix(N, nz) {}

  void fillFromCOO(const MatrixCOO &coo);
};

struct MatrixELL : Matrix {
  int maxNz;
  int elements;

  std::unique_ptr<int[]> length;
  std::unique_ptr<int[]> index;
  std::unique_ptr<floatType[]> data;

  MatrixELL() = delete;
  MatrixELL(int N, int nz, int maxNz) : Matrix(N, nz), maxNz(maxNz) {
    elements = maxNz * N;
  }

  void fillFromCOO(const MatrixCOO &coo);
};

#endif
