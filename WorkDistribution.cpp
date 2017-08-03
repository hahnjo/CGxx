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

#include <cassert>

// #define DEBUG_WORK_DISTRIBUTION
#ifdef DEBUG_WORK_DISTRIBUTION
#include <iostream>
#endif

#include "Matrix.h"
#include "WorkDistribution.h"

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

#ifdef DEBUG_WORK_DISTRIBUTION
    std::cout << "Chunk " << i << " of " << numberOfChunks << " with length "
              << lengths[i] << " from offset " << offsets[i] << std::endl;
#endif

    currentOffset += lengths[i];
  }

  return new WorkDistribution(numberOfChunks, std::move(offsets),
                              std::move(lengths));
}

WorkDistribution *WorkDistribution::calculateByNz(const MatrixCOO &coo,
                                                  int numberOfChunks) {
  std::unique_ptr<int[]> offsets(new int[numberOfChunks]);
  std::unique_ptr<int[]> lengths(new int[numberOfChunks]);

  int chunkLengthInNz = coo.nz / numberOfChunks;
  int remainderInNz = coo.nz - chunkLengthInNz * numberOfChunks;

  int currentOffset = 0;
  int currentNz = 0;
#ifdef DEBUG_WORK_DISTRIBUTION
  int lastNz = 0;
#endif
  for (int i = 0; i < numberOfChunks; i++) {
    offsets[i] = currentOffset;

    int chunkEndInNz = (i + 1) * chunkLengthInNz;
    if ((i + 1) < remainderInNz) {
      chunkEndInNz += i + 1;
    } else {
      chunkEndInNz += remainderInNz;
    }

    while (currentNz < chunkEndInNz) {
      // We might go behind chunkEndInNz here. This is ok because larger chunks
      // in the beginning may hide the start of later chunks!
      currentNz += coo.nzPerRow[currentOffset];
      currentOffset++;
    }

    lengths[i] = currentOffset - offsets[i];

#ifdef DEBUG_WORK_DISTRIBUTION
    std::cout << "Chunk " << i << " of " << numberOfChunks << " with length "
              << lengths[i] << " (" << (currentNz - lastNz)
              << " nonzeros) from offset " << offsets[i] << std::endl;
    lastNz = currentNz;
#endif
  }
  assert(currentNz == coo.nz);

  return new WorkDistribution(numberOfChunks, std::move(offsets),
                              std::move(lengths));
}
