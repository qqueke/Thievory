#include "cc_kernels.cuh"

// Testar o mesmo metodo com warps que utilizamos pro demand
__global__ void
CC32_Static_Kernel(const uint32 *staticSize, const uint32 *d_staticList,
                   const uint64 *d_offsets, const uint32 *d_staticEdges,
                   uint32 *d_values, bool *d_frontier, const bool *d_inStatic) {
  // for (uint32 index = blockIdx.x * blockDim.x + threadIdx.x;
  //     index < *staticSize; index += blockDim.x * gridDim.x) {
  //  uint32 vertexId = d_staticList[index];

  //  // Pretty sure we can remove this but lets review it first
  //  // if (d_inStatic[vertexId])
  //  // {
  //  // CC specific
  //  uint32 sourceValue = d_values[vertexId];

  //  // Neighbors to access
  //  uint64 startNeighbor = d_offsets[vertexId];
  //  uint64 endNeighbor = d_offsets[vertexId + 1];

  //  for (uint64 i = startNeighbor; i < endNeighbor; i++) {
  //    uint32 neighborId = d_staticEdges[i];

  //    // If this new path has lower cost than the previous then change and add
  //    // the neighbor to the frontier
  //    if (sourceValue < d_values[neighborId]) {
  //      atomicMin(&d_values[neighborId], sourceValue);
  //      d_frontier[neighborId] = 1;
  //    }
  //  }
  //  // }
  //}

  // (Row) + (Column) + (Thread Offset)
  const uint32 tid = blockDim.x * THREADS_PER_BLOCK * blockIdx.y +
                     blockDim.x * blockIdx.x + threadIdx.x;
  uint32 warpIdx = tid >> WARP_SHIFT;
  const uint32 laneIdx = tid & ((1 << WARP_SHIFT) - 1);
  const uint32 numWarps = gridDim.x * gridDim.y * THREADS_PER_BLOCK / WARP_SIZE;

  // Grid-Stride loop using Warp ID makes it easier to calculate with the .y
  // dimension
  for (; warpIdx < *staticSize; warpIdx += numWarps) {
    const uint32 traverseIndex = warpIdx;
    uint32 vertexId = d_staticList[traverseIndex];

    uint32 sourceValue = d_values[vertexId];

    const uint64 start = d_offsets[vertexId];
    const uint64 shiftStart = start & MEM_ALIGN_32;
    const uint64 end = d_offsets[vertexId + 1];

    for (uint64 i = shiftStart + laneIdx; i < end; i += WARP_SIZE) {
      if (i >= start) {
        uint32 neighborId = d_staticEdges[i];

        // If this new path has lower cost than the previous then change and add
        // the neighbor to the frontier
        if (sourceValue < d_values[neighborId]) {
          atomicMin(&d_values[neighborId], sourceValue);
          d_frontier[neighborId] = 1;
        }
      }
    }
  }
}

__global__ void
CC64_Static_Kernel(const uint64 *staticSize, const uint64 *d_staticList,
                   const uint64 *d_offsets, const uint64 *d_staticEdges,
                   uint64 *d_values, bool *d_frontier, const bool *d_inStatic) {
  for (uint32 index = blockIdx.x * blockDim.x + threadIdx.x;
       index < *staticSize; index += blockDim.x * gridDim.x) {
    uint64 vertexId = d_staticList[index];

    // Pretty sure we can remove this but lets review it first
    if (d_inStatic[vertexId]) {
      // BFS specific
      uint64 sourceValue = d_values[vertexId];

      // Neighbors to access
      uint64 startNeighbor = d_offsets[vertexId];
      uint64 endNeighbor = d_offsets[vertexId + 1];

      for (uint64 i = startNeighbor; i < endNeighbor; i++) {
        uint64 neighborId = d_staticEdges[i];

        // If this new path has lower cost than the previous then change and add
        // the neighbor to the frontier
        if (sourceValue < d_values[neighborId]) {
          atomicMin(&d_values[neighborId], sourceValue);
          d_frontier[neighborId] = 1;
        }
      }
    }
  }
}

__global__ void CC32_Demand_Kernel(const uint32 *demandSize,
                                   const uint32 *d_demandList, uint32 *d_values,
                                   bool *d_frontier, const uint32 *h_edges,
                                   const uint64 *d_offsets) {
  // (Row) + (Column) + (Thread Offset)
  const uint32 tid = blockDim.x * THREADS_PER_BLOCK * blockIdx.y +
                     blockDim.x * blockIdx.x + threadIdx.x;
  uint32 warpIdx = tid >> WARP_SHIFT;
  const uint32 laneIdx = tid & ((1 << WARP_SHIFT) - 1);
  const uint32 numWarps = gridDim.x * gridDim.y * THREADS_PER_BLOCK / WARP_SIZE;

  // Grid-Stride loop using Warp ID makes it easier to calculate with the .y
  // dimension
  for (; warpIdx < *demandSize; warpIdx += numWarps) {
    const uint32 traverseIndex = warpIdx;
    uint32 vertexId = d_demandList[traverseIndex];

    uint32 sourceValue = d_values[vertexId];

    const uint64 start = d_offsets[vertexId];
    const uint64 shiftStart = start & MEM_ALIGN_32;
    const uint64 end = d_offsets[vertexId + 1];

    for (uint64 i = shiftStart + laneIdx; i < end; i += WARP_SIZE) {
      if (i >= start) {
        uint32 neighborId = h_edges[i];

        // If this new path has lower cost than the previous then change and add
        // the neighbor to the frontier
        if (sourceValue < d_values[neighborId]) {
          atomicMin(&d_values[neighborId], sourceValue);
          d_frontier[neighborId] = 1;
        }
      }
    }
  }
}

__global__ void CC64_Demand_Kernel(const uint64 *demandSize,
                                   const uint64 *d_demandList, uint64 *d_values,
                                   bool *d_frontier, const uint64 *h_edges,
                                   const uint64 *d_offsets) {
  // (Row) + (Column) + (Thread Offset)
  const uint32 tid = blockDim.x * THREADS_PER_BLOCK * blockIdx.y +
                     blockDim.x * blockIdx.x + threadIdx.x;
  uint32 warpIdx = tid >> WARP_SHIFT;
  const uint32 laneIdx = tid & ((1 << WARP_SHIFT) - 1);
  const uint32 numWarps = gridDim.x * gridDim.y * THREADS_PER_BLOCK / WARP_SIZE;

  // Grid-Stride loop using Warp ID makes it easier to calculate with the .y
  // dimension
  for (; warpIdx < *demandSize; warpIdx += numWarps) {
    const uint32 traverseIndex = warpIdx;
    uint64 vertexId = d_demandList[traverseIndex];

    // uint64 srcValue = d_values[id];
    uint64 sourceValue = d_values[vertexId];

    const uint64 start = d_offsets[vertexId];
    const uint64 shiftStart = start & MEM_ALIGN_64;
    const uint64 end = d_offsets[vertexId + 1];

    for (uint64 i = shiftStart + laneIdx; i < end; i += WARP_SIZE) {
      if (i >= start) {
        uint64 neighborId = h_edges[i];

        // If this new path has lower cost than the previous then change and add
        // the neighbor to the frontier
        if (sourceValue < d_values[neighborId]) {
          atomicMin(&d_values[neighborId], sourceValue);
          d_frontier[neighborId] = 1;
        }
      }
    }
  }
}
