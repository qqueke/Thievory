#ifndef SSSP_H
#define SSSP_H

#include "../graph.cu"
#include "../common.cuh"
#include "../timer.cuh"
#include "sssp_kernels.cuh"

void SSSP32(string filePath, uint32 srcVertex, double memAdvise, uint32 nRuns);
void SSSP64(string filePath, uint32 srcVertex, double memAdvise, uint32 nRuns);

#endif
