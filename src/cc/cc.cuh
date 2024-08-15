#ifndef CC_H
#define CC_H

#include "../common.cuh"
#include "../graph.cuh"
#include "../timer.cuh"
#include "cc_kernels.cuh"

void CC32(string filePath, uint32 nRuns, uint32 nNeighborGPUs);
void CC64(string filePath, uint32 nRuns);

#endif
