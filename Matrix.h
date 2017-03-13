#ifndef MATRIX_H
#define MATRIX_H

#include <memory>

#include "def.h"

struct MatrixCOO {
  int N;
  int nz;

  std::unique_ptr<int[]> I;
  std::unique_ptr<int[]> J;
  std::unique_ptr<floatType[]> V;

  std::unique_ptr<int[]> nzPerRow;

  void readFromFile(const char *file);
};

#endif
