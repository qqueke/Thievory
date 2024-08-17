#ifndef SSSP_H
#define SSSP_H

#include "../common.cuh"
#include "../graph.cuh"
#include "sssp_kernels.cuh"

void SSSP32(std::string filePath, uint32 srcVertex, uint32 nRuns,
            uint32 nNeighborGPUs);
void SSSP64(std::string filePath, uint32 srcVertex, uint32 nRuns);

#endif
