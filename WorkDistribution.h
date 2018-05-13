// SPDX-License-Identifier:	GPL-3.0-or-later

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
