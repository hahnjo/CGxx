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

#ifndef WORK_DISTRIBUTION_H
#define WORK_DISTRIBUTION_H

#include <memory>

// Forward declaration to not include Matrix.h
struct MatrixCOO;

/// Chunks for distributing the work.
struct WorkDistribution {
  /// Number of chunks represented by this structure.
  int numberOfChunks;
  /// Offsets of chunks.
  std::unique_ptr<int[]> offsets;
  /// Lengths of chunks.
  std::unique_ptr<int[]> lengths;

  /// Fill structure with given data.
  WorkDistribution(int numberOfChunks, std::unique_ptr<int[]> &&offsets,
                   std::unique_ptr<int[]> &&lengths)
      : numberOfChunks(numberOfChunks), offsets(std::move(offsets)),
        lengths(std::move(lengths)) {}

  /// @return chunk that contains \a row.
  int findChunk(int row) const;

  /// @return true if \a column is on the diagonal of \a chunk.
  bool isOnDiagonal(int chunk, int column) const {
    return offsets[chunk] <= column && column < offsets[chunk] + lengths[chunk];
  }

  /// @return a distribution where each chunk received roughly the same number
  /// of rows.
  static WorkDistribution *calculateByRow(int N, int numberOfChunks);

  /// @return a distribution where each chunk received roughly the same amount
  /// of nonzeros.
  static WorkDistribution *calculateByNz(const MatrixCOO &coo,
                                         int numberOfChunks);
};

#endif
