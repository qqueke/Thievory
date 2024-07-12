#ifndef PR_H
#define PR_H

#include "../graph.cu"
#include "../common.cuh"
#include "../timer.cuh"
#include "pr_kernels.cuh"

void PR32(string filePath, double memAdvise, uint32 nRuns);
void PR64(string filePath, double memAdvise, uint32 nRuns);

#endif
