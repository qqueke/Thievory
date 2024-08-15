#ifndef SSSP_H
#define SSSP_H

#include "../common.cuh"
#include "../graph.cuh"
#include "../timer.cuh"
#include "sssp_kernels.cuh"

void SSSP32(string filePath, uint32 srcVertex, uint32 nRuns,
            uint32 nNeighborGPUs);
void SSSP64(string filePath, uint32 srcVertex, uint32 nRuns);

#endif
