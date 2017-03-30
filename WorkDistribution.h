#ifndef WORK_DISTRIBUTION_H
#define WORK_DISTRIBUTION_H

#include <memory>

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

  /// @return a distribution where each chunk received roughly the same number
  /// of rows.
  static WorkDistribution *calculateByRow(int N, int numberOfChunks);
};

#endif