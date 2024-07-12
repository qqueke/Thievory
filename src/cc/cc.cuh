#ifndef CC_H
#define CC_H

#include "../graph.cuh"
#include "../common.cuh"
#include "../timer.cuh"
#include "cc_kernels.cuh"

void CC32(string filePath, double memAdvise, uint32 nRuns);
void CC64(string filePath, double memAdvise, uint32 nRuns);

#endif
