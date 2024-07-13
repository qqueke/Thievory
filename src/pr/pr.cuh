#ifndef PR_H
#define PR_H

#include "../common.cuh"
#include "../graph.cu"
#include "../timer.cuh"
#include "pr_kernels.cuh"

void PR32(string filePath, double memAdvise, uint32 nRuns,
          uint32 nNeighborGPUs);
void PR64(string filePath, double memAdvise, uint32 nRuns);

#endif
