#ifndef BFS_H
#define BFS_H

#include "../common.cuh"
#include "../graph.cuh"
#include "../timer.cuh"
#include "bfs_kernels.cuh"

void BFS32(string filePath, uint32 srcVertex, double memAdvise, uint32 nRuns,
           uint32 nNeighborGPUs);

void BFS64(string filePath, uint32 srcVertex, double memAdvise, uint32 nRuns);

#endif
