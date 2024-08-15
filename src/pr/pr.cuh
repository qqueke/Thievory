#ifndef PR_H
#define PR_H

#include "../common.cuh"
#include "../graph.cuh"
#include "../timer.cuh"
#include "pr_kernels.cuh"

void PR32(string filePath, uint32 nRuns, uint32 nNeighborGPUs);
void PR64(string filePath, uint32 nRuns);

void PR32_PUSH(string filePath, uint32 nRuns, uint32 nNeighborGPUs);

#endif
