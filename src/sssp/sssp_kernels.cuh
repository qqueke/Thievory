#ifndef SSSP_KERNELS_H
#define SSSP_KERNELS_H

#include "../common.cuh"

__global__ void
SSSP32_Static_Kernel(const uint32 *h_numVertices, const uint32 *d_staticList,
                     const uint64 *d_offsets, const uint32 *d_staticEdges,
                     const uint32 *d_staticWeights, uint32 *d_values,
                     bool *d_frontier, bool *d_staticFrontier);

__global__ void
SSSP64_Static_Kernel(const uint64 *h_numVertices, const uint64 *d_staticList,
                     const uint64 *d_offsets, const uint64 *d_staticEdges,
                     const uint64 *d_staticWeights, uint64 *d_values,
                     bool *d_frontier, const bool *d_inStatic);

__global__ void
SSSP32_Demand_Kernel(const uint32 *demandSize, const uint32 *d_demandList,
                     uint32 *d_values, bool *d_frontier, const uint32 *h_edges,
                     const uint32 *h_weights, const uint64 *d_offsets);

__global__ void
SSSP64_Demand_Kernel(const uint64 *demandSize, const uint64 *d_demandList,
                     uint64 *d_values, bool *d_frontier, const uint64 *h_edges,
                     const uint64 *h_weights, const uint64 *d_offsets);

__global__ void
SSSP32_Filter_Kernel(const uint32 *partitionList, uint32 *d_partitionsOffsets,
                     uint32 *d_values, bool *d_frontier,
                     const uint32 *d_filterEdges, const uint64 *d_offsets,
                     bool *d_filterFrontier, const uint32 *d_filterWeights);

__global__ void SSSP32_NeighborFilter_Kernel(
    const uint32 *partitionList, uint32 *d_partitionsOffsets, uint32 *d_values,
    bool *d_frontier, const uint32 *d_filterEdges, const uint64 *d_offsets,
    bool *d_filterFrontier, const uint32 *d_filterWeights);

#endif
