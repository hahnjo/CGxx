#include <cassert>

#include "WorkDistribution.h"

WorkDistribution *WorkDistribution::calculateByRow(int N, int numberOfChunks) {
  std::unique_ptr<int[]> offsets(new int[numberOfChunks]);
  std::unique_ptr<int[]> lengths(new int[numberOfChunks]);

  int chunkLength = N / numberOfChunks;
  int remainder = N - chunkLength * numberOfChunks;

  int currentOffset = 0;
  for (int i = 0; i < numberOfChunks; i++) {
    offsets[i] = currentOffset;
    lengths[i] = chunkLength;
    if (i < remainder) {
      lengths[i]++;
    }
    currentOffset += lengths[i];
  }

  return new WorkDistribution(numberOfChunks, std::move(offsets),
                              std::move(lengths));
}

int WorkDistribution::findChunk(int row) const {
  // Find corresponding chunk, linear search should be fine here...
  for (int chunk = 0; chunk < numberOfChunks; chunk++) {
    if (row < offsets[chunk] + lengths[chunk]) {
      return chunk;
    }
  }

  assert(0 && "Should have found a chunk!");
  return -1;
}
