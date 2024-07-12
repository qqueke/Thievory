#include "cc.cuh"
#include <iostream>
#include <ostream>

#include <numa.h>
#include <queue>
#include <vector>

#define N_FILTER_STREAMS2 128

__global__ void CalculateCostNSplitFrontiers15(const uint32 *demandSize,
                                               uint32 *d_values,
                                               bool *d_frontier,
                                               const uint32 *h_edges,
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
    // if (!d_frontier[warpIdx]) {
    //   continue;
    // }
    uint32 touch;

    uint32 sourceValue = d_values[warpIdx];
    const uint64 start = d_offsets[warpIdx];
    const uint64 shiftStart = start & MEM_ALIGN_32;
    const uint64 end = d_offsets[warpIdx + 1];

    for (uint64 i = shiftStart + laneIdx; i < end; i += WARP_SIZE) {
      if (i >= start) {

        // uint32 neighborId = h_edges[i];
        //
        // if (sourceValue < d_values[neighborId]) {
        //   // atomicMin(&d_values[neighborId], sourceValue);
        //   //  d_frontier[neighborId] = 1;
        //   d_values[neighborId] = sourceValue;
        // }
        touch = h_edges[i];
        d_values[touch] = sourceValue;
      }
    }
  }
}

__global__ void CalculateCostNSplitFrontiers14(
    const uint32 *partitionList, uint32 *d_partitionsOffsets, uint32 *d_values,
    bool *d_frontier, const uint32 *d_filterEdges, const uint64 *d_offsets) {
  // (Row) + (Column) + (Thread Offset)
  const uint32 tid = blockDim.x * THREADS_PER_BLOCK * blockIdx.y +
                     blockDim.x * blockIdx.x + threadIdx.x;
  uint32 warpIdx = tid >> 10;
  const uint32 laneIdx = tid & ((1 << 10) - 1);
  const uint32 numWarps = gridDim.x * gridDim.y * THREADS_PER_BLOCK / 1024;

  // Grid-Stride loop using Warp ID makes it easier to calculate with the .y
  // dimension
  for (; warpIdx < N_FILTER_STREAMS2; warpIdx += numWarps) {

    uint32 partition = partitionList[warpIdx];

    // Start offset
    // d_partitionsOffsets[partition]

    // End offset
    // d_partitionsOffsets[partition + 1]
    uint32 touch;
    // Start Edge
    uint32 startEdge = d_offsets[d_partitionsOffsets[partition]];

    // End Edge
    uint32 endEdge = d_offsets[d_partitionsOffsets[partition + 1]];

    uint32 edgeCount = endEdge - startEdge;

    //  if (!d_frontier[warpIdx]) {
    //    continue;
    //  }

    uint32 sourceValue = d_values[d_partitionsOffsets[partition]];

    const uint64 start = warpIdx * EDGES_IN_PARTITION;
    const uint64 end = start + edgeCount;

    for (uint64 i = start + laneIdx; i < end; i += WARP_SIZE) {

      // d_values[warpIdx] += h_edges[warpIdx];

      touch = d_filterEdges[i + warpIdx * EDGES_IN_PARTITION];
      d_values[touch] = sourceValue;
    }
  }
}

__global__ void CalculateCostNSplitFrontiers11(
    const uint32 *partitionList, uint32 *d_partitionsOffsets, uint32 *d_values,
    bool *d_frontier, const uint32 *d_filterEdges, const uint64 *d_offsets) {

  const uint32 tid = blockIdx.x * blockDim.x + threadIdx.x;
  // const uint32 tid = blockDim.x * THREADS_PER_BLOCK * blockIdx.y +
  //                    blockDim.x * blockIdx.x + threadIdx.x;

  uint32 warpIdx = tid >> WARP_SHIFT;
  const uint32 laneIdx = tid & ((1 << WARP_SHIFT) - 1);
  const uint32 numWarps = gridDim.x * gridDim.y * THREADS_PER_BLOCK / WARP_SIZE;

  uint32 partition = partitionList[0];
  uint32 touch;
  //  Start Edge
  uint32 startEdge = d_offsets[d_partitionsOffsets[partition]];

  for (warpIdx += d_partitionsOffsets[partition];
       warpIdx < d_partitionsOffsets[partition + 1]; warpIdx += numWarps) {

    uint32 sourceValue = d_values[warpIdx];

    const uint64 start = d_offsets[warpIdx] - startEdge;
    const uint64 end = d_offsets[warpIdx + 1] - startEdge;

    for (uint64 i = start + laneIdx; i < end; i += WARP_SIZE) {
      touch = d_filterEdges[i];
      d_values[touch] = sourceValue;
    }
  }
}

__global__ void CalculateCostNSplitFrontiers16(const uint32 *demandSize,
                                               uint32 *d_values,
                                               bool *d_frontier,
                                               const uint32 *d_edges,
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
    // if (!d_frontier[warpIdx]) {
    //   continue;
    // }
    uint32 touch;

    uint32 sourceValue = d_values[warpIdx];
    const uint64 start = d_offsets[warpIdx];
    const uint64 end = d_offsets[warpIdx + 1];

    for (uint64 i = start + laneIdx; i < end; i += WARP_SIZE) {

      // uint32 neighborId = h_edges[i];
      //
      // if (sourceValue < d_values[neighborId]) {
      //   // atomicMin(&d_values[neighborId], sourceValue);
      //   //  d_frontier[neighborId] = 1;
      //   d_values[neighborId] = sourceValue;
      // }
      touch = d_edges[i];
      d_values[touch] = sourceValue;
    }
  }
}

__global__ void CalculateCostNSplitFrontiers17(
    const uint32 *partitionList, uint32 *d_partitionsOffsets, uint32 *d_values,
    bool *d_frontier, const uint32 *d_filterEdges, const uint64 *d_offsets) {
  // (Row) + (Column) + (Thread Offset)
  const uint32 tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32 warpIdx = tid >> WARP_SHIFT;
  const uint32 laneIdx = tid & ((1 << WARP_SHIFT) - 1);
  const uint32 numWarps = gridDim.x * gridDim.y * THREADS_PER_BLOCK / WARP_SIZE;

  // Grid-Stride loop using Warp ID makes it easier to calculate with the .y
  // dimension
  for (; warpIdx < N_FILTER_STREAMS2; warpIdx += numWarps) {

    uint32 partition = partitionList[warpIdx];

    // Start offset
    // d_partitionsOffsets[partition]

    // End offset
    // d_partitionsOffsets[partition + 1]
    uint32 touch;

    // Start Edge
    uint32 startEdge = d_offsets[d_partitionsOffsets[partition]];

    // End Edge
    uint32 endEdge = d_offsets[d_partitionsOffsets[partition + 1]];

    uint32 edgeCount = endEdge - startEdge;

    //  if (!d_frontier[warpIdx]) {
    //    continue;
    //  }

    uint32 sourceValue = d_values[d_partitionsOffsets[partition]];

    const uint64 start = warpIdx * EDGES_IN_PARTITION;
    const uint64 end = start + edgeCount;

    for (uint64 i = start + laneIdx; i < end; i += WARP_SIZE) {

      // d_values[warpIdx] += h_edges[warpIdx];

      touch = d_filterEdges[i + warpIdx * EDGES_IN_PARTITION];
      d_values[touch] = sourceValue;
    }
  }
}

__global__ void CalculateCostNSplitFrontiers20(
    const uint32 *partitionList, uint32 *d_partitionsOffsets, uint32 *d_values,
    bool *d_frontier, const uint32 *d_filterEdges, const uint64 *d_offsets,
    bool *d_filterFrontier) {

  const uint32 tid = blockIdx.x * blockDim.x + threadIdx.x;
  // const uint32 tid = blockDim.x * THREADS_PER_BLOCK * blockIdx.y +
  //                    blockDim.x * blockIdx.x + threadIdx.x;

  uint32 warpIdx = tid >> WARP_SHIFT;
  const uint32 laneIdx = tid & ((1 << WARP_SHIFT) - 1);
  const uint32 numWarps = gridDim.x * gridDim.y * THREADS_PER_BLOCK / WARP_SIZE;

  uint32 partition = partitionList[0];
  // uint32 touch;
  //  Start Edge
  uint32 startEdge = d_offsets[d_partitionsOffsets[partition]];

  // End Edge
  // uint32 endEdge = d_offsets[d_partitionsOffsets[partition + 1]];

  // uint32 edgeCount = endEdge - startEdge;

  // Grid-Stride loop using Warp ID makes it easier to calculate with the .y
  // dimension
  for (warpIdx += d_partitionsOffsets[partition];
       warpIdx < d_partitionsOffsets[partition + 1]; warpIdx += numWarps) {
    // Start offset
    // d_partitionsOffsets[partition]

    // End offset
    // d_partitionsOffsets[partition + 1]

    if (!d_filterFrontier[warpIdx])
      continue;

    d_filterFrontier[warpIdx] = 0;

    uint32 sourceValue = d_values[warpIdx];

    const uint64 start = d_offsets[warpIdx] - startEdge;
    const uint64 end = d_offsets[warpIdx + 1] - startEdge;

    for (uint64 i = start + laneIdx; i < end; i += WARP_SIZE) {
      uint32 neighborId = d_filterEdges[i];

      // If this new path has lower cost than the previous then change and add
      // the neighbor to the frontier
      if (sourceValue < d_values[neighborId]) {
        atomicMin(&d_values[neighborId], sourceValue);
        d_frontier[neighborId] = 1;
      }
    }
  }

  //  uint32 tid = blockIdx.x * blockDim.x + threadIdx.x;
  //
  //  uint32 partition = partitionList[0];
  //  // uint32 touch;
  //  //  Start Edge
  //  uint32 startEdge = d_offsets[d_partitionsOffsets[partition]];
  //
  //  for (tid += d_partitionsOffsets[partition];
  //       tid < d_partitionsOffsets[partition + 1];
  //       tid += blockDim.x * gridDim.x) {
  //
  //
  //    if (!d_filterFrontier[tid])
  //      continue;
  //
  //    d_filterFrontier[tid] = 0;
  //
  //    uint32 sourceValue = d_values[tid];
  //
  //    const uint64 start = d_offsets[tid] - startEdge;
  //    const uint64 end = d_offsets[tid + 1] - startEdge;
  //
  //    for (uint64 i = start; i < end; i ++ ) {
  //      uint32 neighborId = d_filterEdges[i];
  //
  //      // If this new path has lower cost than the previous then change and
  //      add
  //      // the neighbor to the frontier
  //      if (sourceValue < d_values[neighborId]) {
  //        atomicMin(&d_values[neighborId], sourceValue);
  //        d_frontier[neighborId] = 1;
  //      }
  //    }
  //  }
}

__global__ void CalculateCostNSplitFrontiers21(
    const uint32 *partitionList, uint32 *d_partitionsOffsets, uint32 *d_values,
    bool *d_frontier, const uint32 *d_filterEdges, const uint64 *d_offsets,
    bool *d_filterFrontier) {

  const uint32 tid = blockIdx.x * blockDim.x + threadIdx.x;
  // const uint32 tid = blockDim.x * THREADS_PER_BLOCK * blockIdx.y +
  //                    blockDim.x * blockIdx.x + threadIdx.x;

  uint32 warpIdx = tid >> WARP_SHIFT;
  const uint32 laneIdx = tid & ((1 << WARP_SHIFT) - 1);
  const uint32 numWarps = gridDim.x * gridDim.y * THREADS_PER_BLOCK / WARP_SIZE;

  uint32 partition = partitionList[0];
  // uint32 touch;
  //  Start Edge
  uint32 startEdge = d_offsets[d_partitionsOffsets[partition]];

  // End Edge
  // uint32 endEdge = d_offsets[d_partitionsOffsets[partition + 1]];

  // uint32 edgeCount = endEdge - startEdge;
  // Grid-Stride loop using Warp ID makes it easier to calculate with the .y
  // dimension
  for (warpIdx += d_partitionsOffsets[partition];
       warpIdx < d_partitionsOffsets[partition + 1]; warpIdx += numWarps) {
    // Start offset
    // d_partitionsOffsets[partition]

    // End offset
    // d_partitionsOffsets[partition + 1]

    if (!d_filterFrontier[warpIdx])
      continue;

    d_filterFrontier[warpIdx] = 0;

    uint32 sourceValue = d_values[warpIdx];

    const uint64 start = d_offsets[warpIdx] - startEdge;
    const uint64 end = d_offsets[warpIdx + 1] - startEdge;

    for (uint64 i = start + laneIdx; i < end; i += WARP_SIZE) {
      uint32 neighborId = d_filterEdges[i];

      // If this new path has lower cost than the previous then change and add
      // the neighbor to the frontier
      if (sourceValue < d_values[neighborId]) {
        atomicMin(&d_values[neighborId], sourceValue);
        d_frontier[neighborId] = 1;
      }
    }
  }
}

__global__ void CalculateCostNSplitFrontiers22(
    const uint32 *partitionList, uint32 *d_partitionsOffsets, uint32 *d_values,
    bool *d_frontier, const uint32 *d_filterEdges, const uint64 *d_offsets,
    bool *d_filterFrontier) {

  const uint32 tid = blockIdx.x * blockDim.x + threadIdx.x;
  // const uint32 tid = blockDim.x * THREADS_PER_BLOCK * blockIdx.y +
  //                    blockDim.x * blockIdx.x + threadIdx.x;

  uint32 warpIdx = tid >> WARP_SHIFT;
  const uint32 laneIdx = tid & ((1 << WARP_SHIFT) - 1);
  const uint32 numWarps = gridDim.x * gridDim.y * THREADS_PER_BLOCK / WARP_SIZE;

  uint32 partition = partitionList[0];
  // uint32 touch;
  //  Start Edge
  uint32 startEdge = d_offsets[d_partitionsOffsets[partition]];

  // End Edge
  // uint32 endEdge = d_offsets[d_partitionsOffsets[partition + 1]];

  // uint32 edgeCount = endEdge - startEdge;
  // Grid-Stride loop using Warp ID makes it easier to calculate with the .y
  // dimension
  for (warpIdx += d_partitionsOffsets[partition];
       warpIdx < d_partitionsOffsets[partition + 1]; warpIdx += numWarps) {
    // Start offset
    // d_partitionsOffsets[partition]

    // End offset
    // d_partitionsOffsets[partition + 1]

    if (!d_filterFrontier[warpIdx])
      continue;

    d_filterFrontier[warpIdx] = 0;

    uint32 sourceValue = d_values[warpIdx];

    const uint64 start = d_offsets[warpIdx] - startEdge;
    const uint64 end = d_offsets[warpIdx + 1] - startEdge;

    for (uint64 i = start + laneIdx; i < end; i += WARP_SIZE) {
      uint32 neighborId = d_filterEdges[i];

      // If this new path has lower cost than the previous then change and add
      // the neighbor to the frontier
      if (sourceValue < d_values[neighborId]) {
        atomicMin(&d_values[neighborId], sourceValue);
        d_frontier[neighborId] = 1;
      }
    }
  }
}
__global__ void CalculateCostNSplitFrontiers23(
    const uint32 *partitionList, uint32 *d_partitionsOffsets, uint32 *d_values,
    bool *d_frontier, const uint32 *d_filterEdges, const uint64 *d_offsets,
    bool *d_filterFrontier) {

  const uint32 tid = blockIdx.x * blockDim.x + threadIdx.x;
  // const uint32 tid = blockDim.x * THREADS_PER_BLOCK * blockIdx.y +
  //                    blockDim.x * blockIdx.x + threadIdx.x;

  uint32 warpIdx = tid >> WARP_SHIFT;
  const uint32 laneIdx = tid & ((1 << WARP_SHIFT) - 1);
  const uint32 numWarps = gridDim.x * gridDim.y * THREADS_PER_BLOCK / WARP_SIZE;

  uint32 partition = partitionList[0];
  // uint32 touch;
  //  Start Edge
  uint32 startEdge = d_offsets[d_partitionsOffsets[partition]];

  // End Edge
  // uint32 endEdge = d_offsets[d_partitionsOffsets[partition + 1]];

  // uint32 edgeCount = endEdge - startEdge;
  // Grid-Stride loop using Warp ID makes it easier to calculate with the .y
  // dimension
  for (warpIdx += d_partitionsOffsets[partition];
       warpIdx < d_partitionsOffsets[partition + 1]; warpIdx += numWarps) {
    // Start offset
    // d_partitionsOffsets[partition]

    // End offset
    // d_partitionsOffsets[partition + 1]

    if (!d_filterFrontier[warpIdx])
      continue;

    d_filterFrontier[warpIdx] = 0;

    uint32 sourceValue = d_values[warpIdx];

    const uint64 start = d_offsets[warpIdx] - startEdge;
    const uint64 end = d_offsets[warpIdx + 1] - startEdge;

    for (uint64 i = start + laneIdx; i < end; i += WARP_SIZE) {
      uint32 neighborId = d_filterEdges[i];

      // If this new path has lower cost than the previous then change and add
      // the neighbor to the frontier
      if (sourceValue < d_values[neighborId]) {
        atomicMin(&d_values[neighborId], sourceValue);
        d_frontier[neighborId] = 1;
      }
    }
  }
}

// #define ZC

void CC32(string filePath, double memAdvise, uint32 nRuns,
          uint32 nNeighborGPUs) {
  numa_run_on_node(0);
  ALGORITHM_TYPE algo = CC;
  CSR<uint32> *graph = new CSR<uint32>;
  graph->ReadInputFile(filePath, algo);
  graph->InitData(0, nNeighborGPUs);

  // Adjust this number of blocks in x dimension to be a multiple of the number
  // of SMS and acquire better load balancing
  int device = 0;
  uint32 k = 4;

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);
  uint32 numSMs = prop.multiProcessorCount;

  dim3 staticGrid = dim3(k * numSMs, 1, 1);
  dim3 blockDim(THREADS_PER_BLOCK, 1, 1); // (x,y,z) = (THREADS_PER_BLOCK, 1, 1)

  uint32 totalParts = 0;

  cudaStream_t staticStream, demandStream, frontierStream;

  GPUAssert(cudaStreamCreate(&frontierStream));
  GPUAssert(cudaStreamCreate(&staticStream));
  GPUAssert(cudaStreamCreate(&demandStream));

  auto asyncFrontierPolicy = thrust::cuda::par_nosync.on(frontierStream);
  auto asyncStaticPolicy = thrust::cuda::par_nosync.on(staticStream);
  auto asyncDemandPolicy = thrust::cuda::par_nosync.on(demandStream);
  // auto syncPolicy  = thrust::cuda::par.on(staticStream);

  auto syncFrontierPolicy = thrust::cuda::par.on(frontierStream);
  auto syncStaticPolicy = thrust::cuda::par.on(staticStream);
  auto syncDemandPolicy = thrust::cuda::par.on(demandStream);

  TimeRecord<chrono::milliseconds> totalProcess("Total execution");

  TimeRecord<chrono::milliseconds> test0("Copy to GPU 0");
  TimeRecord<chrono::milliseconds> test1("Copy to GPU 1");
  TimeRecord<chrono::milliseconds> test2("Copy to GPU 2");
  TimeRecord<chrono::milliseconds> test3("Copy to GPU 3");

  TimeRecord<chrono::milliseconds> k0("Kernel GPU 0");
  TimeRecord<chrono::milliseconds> k1("Kernel GPU 1");
  TimeRecord<chrono::milliseconds> k2("Kernel GPU 2");
  TimeRecord<chrono::milliseconds> k3("Kernel GPU 3");

  uint32 nGPUs = nNeighborGPUs + 1;

  std::vector<std::array<cudaStream_t, N_FILTER_STREAMS2>>
      neighborMemCpyStreams(nNeighborGPUs);

  std::vector<std::array<cudaStream_t, N_FILTER_STREAMS2>>
      neighborComputeStreams(nNeighborGPUs);

  for (int i = 0; i < nNeighborGPUs; ++i) {
    cudaSetDevice(i + 1);
    for (int j = 0; j < N_FILTER_STREAMS2; ++j)
      GPUAssert(cudaStreamCreate(&neighborMemCpyStreams[i][j]));
  }

  cudaSetDevice(0);

  for (int i = 0; i < nNeighborGPUs; ++i) {
    for (int j = 0; j < N_FILTER_STREAMS2; ++j)
      GPUAssert(cudaStreamCreate(&neighborComputeStreams[i][j]));
  }

  cudaStream_t streams[N_FILTER_STREAMS2];

  for (uint32 i = 0; i < N_FILTER_STREAMS2; i++)
    GPUAssert(cudaStreamCreate(&streams[i]));

  // Removing static data
  // cudaMemset(graph->d_inStatic, 0, *(graph->numVertices) * sizeof(bool));

  GPUAssert(
      cudaDeviceEnablePeerAccess(1, 0)); // Enable peer access with device 0

  GPUAssert(cudaDeviceEnablePeerAccess(2, 0));

  GPUAssert(cudaDeviceEnablePeerAccess(3, 0));

  graph->h_edges2 =
      (uint32 *)numa_alloc_onnode(graph->numEdges * sizeof(uint32), 1);

  cudaHostRegister(graph->h_edges2, graph->numEdges * sizeof(uint32),
                   cudaHostRegisterDefault);

  cudaMemcpy(graph->h_edges2, graph->h_edges,
             graph->numEdges * sizeof(*graph->h_edges2), cudaMemcpyHostToHost);

  cudaDeviceSynchronize();

  std::cout << "Starting Traversals" << std::endl;
  for (int test = 0; test < nRuns; test++) {

    graph->ResetFrontierNValues();

    *(graph->frontierSize) = thrust::reduce(
        graph->thrustFrontier, graph->thrustFrontier + *(graph->numVertices), 0,
        thrust::plus<uint32>());

    totalProcess.startRecord();

    while (*(graph->frontierSize)) {

      //  std::cout << "Frontier size: " << *graph->frontierSize << std::endl;
      setStaticNDemandFrontiers<<<staticGrid, blockDim, 0, frontierStream>>>(
          graph->numVertices, graph->d_frontier, graph->d_staticFrontier,
          graph->d_demandFrontier, graph->d_inStatic);

      cudaStreamSynchronize(frontierStream);

      cudaMemsetAsync(graph->d_frontier, 0,
                      *(graph->numVertices) * sizeof(*graph->d_frontier),
                      frontierStream);

      // Calculate the amount of active nodes in GPU memory
      *(graph->staticSize) =
          thrust::reduce(graph->thurstStaticFrontier,
                         graph->thurstStaticFrontier + *(graph->numVertices), 0,
                         thrust::plus<uint32>());

      if (*graph->frontierSize > 20000000) {
        CalculateCostNSplitFrontiers<uint32>
            <<<staticGrid, blockDim, 0, demandStream>>>(
                graph->numPartitions, graph->d_partitionsOffsets,
                graph->d_offsets, graph->d_partitionCost,
                graph->d_demandFrontier, graph->d_filterFrontier);

        CalculateCostNSplitFrontiers2<uint32>
            <<<staticGrid, blockDim, 0, demandStream>>>(
                graph->numPartitions, graph->d_partitionsOffsets,
                graph->d_offsets, graph->d_partitionCost,
                graph->d_demandFrontier, graph->d_filterFrontier);

        CalculateCostNSplitFrontiers3<uint32>
            <<<staticGrid, blockDim, 0, demandStream>>>(
                graph->numPartitions, graph->d_partitionsOffsets,
                graph->d_offsets, graph->d_partitionCost,
                graph->d_demandFrontier, graph->d_filterFrontier);

        cudaStreamSynchronize(demandStream);

        cudaMemcpyAsync(graph->h_partitionCost, graph->d_partitionCost,
                        *graph->numPartitions * sizeof(*graph->h_partitionCost),
                        cudaMemcpyDeviceToHost, streams[0]);

        cudaMemsetAsync(graph->d_partitionCost, 0,
                        *graph->numPartitions * sizeof(*graph->d_partitionCost),
                        streams[0]);
      }

      // Calculate the amount of active vertices on-demand
      *(graph->demandSize) =
          thrust::reduce(graph->thurstDemandFrontier,
                         graph->thurstDemandFrontier + *(graph->numVertices), 0,
                         thrust::plus<uint32>());

      if (*(graph->staticSize) > 0) {

        thrust::exclusive_scan(
            graph->thurstStaticFrontier,
            graph->thurstStaticFrontier + *(graph->numVertices),
            graph->thurstPrefixSum, 0, thrust::plus<uint32>());

        setStaticList<<<staticGrid, blockDim, 0, staticStream>>>(
            graph->numVertices, graph->d_staticList, graph->d_staticFrontier,
            graph->d_prefixSum);

        cudaStreamSynchronize(frontierStream);

        CC32_Static_Kernel<<<staticGrid, blockDim, 0, staticStream>>>(
            graph->staticSize, graph->d_staticList, graph->d_offsets,
            graph->d_staticEdges, graph->d_values, graph->d_frontier,
            graph->d_inStatic);
      }

      if (*(graph->demandSize) > 0) {
        thrust::exclusive_scan(
            graph->thurstDemandFrontier,
            graph->thurstDemandFrontier + *(graph->numVertices),
            graph->thurstPrefixSum, 0, thrust::plus<uint32>());

        setDemandList<<<staticGrid, blockDim, 0, demandStream>>>(
            graph->numVertices, graph->d_demandList, graph->d_demandFrontier,
            graph->d_prefixSum);

        uint32 numBlocks =
            (((*(graph->demandSize)) * WARP_SIZE + THREADS_PER_BLOCK) /
             THREADS_PER_BLOCK);
        dim3 gridDim(THREADS_PER_BLOCK,
                     (numBlocks + THREADS_PER_BLOCK) / THREADS_PER_BLOCK);

        cudaStreamSynchronize(frontierStream);

        CC32_Demand_Kernel<<<gridDim, blockDim, 0, demandStream>>>(
            graph->demandSize, graph->d_demandList, graph->d_values,
            graph->d_frontier, graph->h_edges, graph->d_offsets);
      }

      // if (*graph->frontierSize > 20000000) {

      //  cudaMemcpy(graph->d_demandFrontier, graph->d_filterFrontier,
      //             *graph->numVertices * sizeof(*graph->d_filterFrontier),
      //             cudaMemcpyDeviceToDevice);

      //  cudaDeviceSynchronize();

      //  *(graph->demandSize) =
      //      thrust::reduce(graph->thurstDemandFrontier,
      //                     graph->thurstDemandFrontier +
      //                     *(graph->numVertices), 0, thrust::plus<uint32>());

      //  cudaDeviceSynchronize();
      //   if (*(graph->demandSize) > 0) {
      //     thrust::exclusive_scan(
      //         graph->thurstDemandFrontier,
      //         graph->thurstDemandFrontier + *(graph->numVertices),
      //         graph->thurstPrefixSum, 0, thrust::plus<uint32>());

      //    setDemandList<<<staticGrid, blockDim, 0, demandStream>>>(
      //        graph->numVertices, graph->d_demandList,
      //        graph->d_demandFrontier, graph->d_prefixSum);

      //    uint32 numBlocks =
      //        (((*(graph->demandSize)) * WARP_SIZE + THREADS_PER_BLOCK) /
      //         THREADS_PER_BLOCK);
      //    dim3 gridDim(THREADS_PER_BLOCK,
      //                 (numBlocks + THREADS_PER_BLOCK) / THREADS_PER_BLOCK);

      //    cudaStreamSynchronize(frontierStream);
      //    cudaStreamSynchronize(staticStream);

      //    CC32_Demand_Kernel<<<gridDim, blockDim, 0, demandStream>>>(
      //        graph->demandSize, graph->d_demandList, graph->d_values,
      //        graph->d_frontier, graph->h_edges, graph->d_offsets);
      //  }
      //}

      if (*graph->frontierSize > 20000000) {

        uint32 numParts = 0;
        uint32 numPartsNGPU = 0;

        uint32 numPartsNGPU0 = 0;
        uint32 numPartsNGPU1 = 0;
        uint32 numPartsNGPU2 = 0;
        uint32 numPartsNGPU3 = 0;

        uint32 teoNumParts = 0;

        std::queue<uint32> targetGPUQueue;
        std::vector<std::queue<uint32>> neighborGPUQueues(nNeighborGPUs);

        std::vector<uint32> partitionList;

        cudaStreamSynchronize(streams[0]);

        for (uint32 partition = 0; partition < *graph->numPartitions;
             partition++) {

          if (graph->h_partitionCost[partition] <= FILTER_THRESHOLD)
            continue;

          partitionList.push_back(partition);
        }

        cudaStreamSynchronize(frontierStream);

        for (uint32 index = 0; index < partitionList.size(); index++) {

          uint32 partition = partitionList[index];

          numParts++;

          // Partition edge start
          uint32 start =
              graph->h_offsets[graph->h_partitionsOffsets[partition]];

          uint32 partitionSize =
              graph->h_offsets[graph->h_partitionsOffsets[partition + 1]] -
              start;

          // uint32 stream = partition % N_FILTER_STREAMS2;

          uint32 stream = (index / nGPUs) % N_FILTER_STREAMS2;

          graph->h_partitionList[stream] = partition;

          cudaStreamSynchronize(streams[stream]);

          // cudaDeviceSynchronize();
          test0.startRecord();
          cudaMemcpyAsync(graph->d_filterEdges[stream], &graph->h_edges[start],
                          partitionSize * sizeof(*graph->h_edges),
                          cudaMemcpyHostToDevice, streams[stream]);

          cudaMemcpyAsync(&graph->d_partitionList[stream],
                          &graph->h_partitionList[stream],
                          sizeof(*graph->h_partitionList),
                          cudaMemcpyHostToDevice, streams[stream]);

          //  cudaDeviceSynchronize();
          test0.endRecord();

          targetGPUQueue.push(stream);

          for (uint32 gpu = 0; gpu < neighborGPUQueues.size(); gpu++) {
            if (index + 1 >= partitionList.size())
              break;

            index++;
            partition = partitionList[index];

            teoNumParts++;

            // Partition edge start
            uint32 neighborStart =
                graph->h_offsets[graph->h_partitionsOffsets[partition]];

            uint32 neighborPartitionSize =
                graph->h_offsets[graph->h_partitionsOffsets[partition + 1]] -
                neighborStart;

            uint32 neighborStream = (index / nGPUs) % N_FILTER_STREAMS2;

            graph->h_nPartList[gpu][neighborStream] = partition;

            // Sync compute stream on device 0
            cudaStreamSynchronize(neighborComputeStreams[gpu][neighborStream]);
            // cudaDeviceSynchronize();
            cudaSetDevice(gpu + 1);

            //   cudaDeviceSynchronize();
            test1.startRecord();

            cudaMemcpyAsync(graph->d_nFilterEdges[gpu][neighborStream],
                            (gpu > 0) ? graph->h_edges2 + neighborStart
                                      : graph->h_edges + neighborStart,
                            neighborPartitionSize * sizeof(*graph->h_edges),
                            cudaMemcpyHostToDevice,
                            neighborMemCpyStreams[gpu][neighborStream]);

            // We can prob allocate this data in the other numa node too
            cudaMemcpyAsync(&graph->d_nPartList[gpu][neighborStream],
                            graph->h_nPartList[gpu] + neighborStream,
                            sizeof(*graph->h_neighborPartitionList),
                            cudaMemcpyHostToDevice,
                            neighborMemCpyStreams[gpu][neighborStream]);

            //  cudaDeviceSynchronize();
            test1.endRecord();

            neighborGPUQueues[gpu].push(neighborStream);

            cudaSetDevice(0);
          }

          while (!targetGPUQueue.empty()) {
            uint32 tStream = targetGPUQueue.front();

            cudaError_t streamStatus = cudaStreamQuery(streams[tStream]);

            if (streamStatus == cudaErrorNotReady) {
              if (targetGPUQueue.size() < N_FILTER_STREAMS2)
                continue;
              else
                cudaStreamSynchronize(streams[tStream]);
            }

            numPartsNGPU0++;
            targetGPUQueue.pop();

            //   cudaDeviceSynchronize();
            k0.startRecord();
            CalculateCostNSplitFrontiers20<<<staticGrid, blockDim, 0,
                                             streams[tStream]>>>(
                &graph->d_partitionList[tStream], graph->d_partitionsOffsets,
                graph->d_values, graph->d_frontier,
                graph->d_filterEdges[tStream], graph->d_offsets,
                graph->d_filterFrontier);
            //       cudaDeviceSynchronize();
            k0.endRecord();
          }

          for (uint32 gpu = 0; gpu < neighborGPUQueues.size(); gpu++) {

            while (!neighborGPUQueues[gpu].empty()) {
              uint32 nStream = neighborGPUQueues[gpu].front();

              cudaSetDevice(gpu + 1);
              cudaError_t streamStatus =
                  cudaStreamQuery(neighborMemCpyStreams[gpu][nStream]);

              if (streamStatus == cudaErrorNotReady) {
                if (neighborGPUQueues[gpu].size() < N_FILTER_STREAMS2) {
                  cudaSetDevice(0);
                  continue;
                } else
                  cudaStreamSynchronize(neighborMemCpyStreams[gpu][nStream]);
              }

              cudaSetDevice(0);
              neighborGPUQueues[gpu].pop();

              CalculateCostNSplitFrontiers21<<<
                  staticGrid, blockDim, 0,
                  neighborComputeStreams[gpu][nStream]>>>(
                  &graph->d_nPartList[gpu][nStream], graph->d_partitionsOffsets,
                  graph->d_values, graph->d_frontier,
                  graph->d_nFilterEdges[gpu][nStream], graph->d_offsets,
                  graph->d_filterFrontier);
            }
          }
        }

        while (!targetGPUQueue.empty()) {

          uint32 tStream = targetGPUQueue.front();

          cudaStreamSynchronize(streams[tStream]);
          numPartsNGPU0++;
          targetGPUQueue.pop();

          k0.startRecord();
          CalculateCostNSplitFrontiers20<<<staticGrid, blockDim, 0,
                                           streams[tStream]>>>(
              &graph->d_partitionList[tStream], graph->d_partitionsOffsets,
              graph->d_values, graph->d_frontier, graph->d_filterEdges[tStream],
              graph->d_offsets, graph->d_filterFrontier);
          //   cudaDeviceSynchronize();
          k0.endRecord();
        }

        for (uint32 gpu = 0; gpu < neighborGPUQueues.size(); gpu++) {

          while (!neighborGPUQueues[gpu].empty()) {
            uint32 nStream = neighborGPUQueues[gpu].front();

            cudaSetDevice(gpu + 1);
            cudaError_t streamStatus =
                cudaStreamQuery(neighborMemCpyStreams[gpu][nStream]);

            if (streamStatus == cudaErrorNotReady) {
              if (neighborGPUQueues[gpu].size() < N_FILTER_STREAMS2) {
                cudaSetDevice(0);
                continue;
              } else
                cudaStreamSynchronize(neighborMemCpyStreams[gpu][nStream]);
            }

            cudaSetDevice(0);
            neighborGPUQueues[gpu].pop();

            CalculateCostNSplitFrontiers21<<<
                staticGrid, blockDim, 0,
                neighborComputeStreams[gpu][nStream]>>>(
                &graph->d_nPartList[gpu][nStream], graph->d_partitionsOffsets,
                graph->d_values, graph->d_frontier,
                graph->d_nFilterEdges[gpu][nStream], graph->d_offsets,
                graph->d_filterFrontier);
            //   cudaDeviceSynchronize();
          }
        }

        std::cout << "Partitions to be processed target GPU: " << numParts
                  << std::endl;
        std::cout << "Partitions processed in GPU 0: " << numPartsNGPU0
                  << std::endl;

        std::cout << "Partitions to be processed in neighbor GPUs: "
                  << teoNumParts << std::endl;

        std::cout << "Partitions processed in neighbor GPUs: " << numPartsNGPU
                  << std::endl;
        std::cout << "Partitions processed in neighbor GPU 1: " << numPartsNGPU1
                  << std::endl;
        std::cout << "Partitions processed in neighbor GPU 2: " << numPartsNGPU2
                  << std::endl;

        std::cout << "Partitions processed in neighbor GPU 3: " << numPartsNGPU3
                  << std::endl;

        totalParts += numParts + teoNumParts;
      }
      cudaDeviceSynchronize();
      // GPUAssert(cudaDeviceSynchronize());

      *(graph->frontierSize) = thrust::reduce(
          graph->thrustFrontier, graph->thrustFrontier + *(graph->numVertices),
          0, thrust::plus<uint32>());
    }
  }

  totalProcess.endRecord();
  totalProcess.print();

  test0.print();
  test1.print();
  test2.print();
  test3.print();

  k0.print();
  k1.print();
  k2.print();
  k3.print();

  uint64 MBytes = totalParts * 16;

  uint64 GBytes = MBytes >> 10;
  std::cout << "Total partitions in filter: " << totalParts << std::endl;

  std::cout << "Total amount of data sent with filter: " << MBytes << " MB"
            << std::endl;
  std::cout << "Total amount of data sent with filter: " << GBytes << " GB"
            << std::endl; // We're gonna need to compare results now!!
  cudaMemcpy(graph->h_values, graph->d_values,
             *(graph->numVertices) * sizeof(uint32), cudaMemcpyDeviceToHost);

  graph->DumpValues();
  return;
}

void CC64(string filePath, double memAdvise, uint32 nRuns) {
  ALGORITHM_TYPE algo = CC;
  CSR<uint32> *graph = new CSR<uint32>;
  graph->ReadInputFile(filePath, algo);
  // graph->InitData(0, 0);
  // Adjust this number of blocks in x dimension to be a multiple of the
  // number of SMS and acquire better load balancing
  int device = 0; // Selected device
  uint32 k =
      4; // Multiple of SMs to choose for the grid dimension (to be adjusted)

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);
  uint32 numSMs = prop.multiProcessorCount;

  dim3 staticGrid = dim3(k * numSMs, 1, 1);
  dim3 blockDim(THREADS_PER_BLOCK, 1,
                1); // (x,y,z) = (THREADS_PER_BLOCK, 1, 1)

  cudaStream_t streams[16];

  for (uint32 i = 0; i < 16; i++)
    GPUAssert(cudaStreamCreate(&streams[i]));

  TimeRecord<chrono::milliseconds> totalProcess("Total execution");

  // Removing static data
  // cudaMemset(graph->d_inStatic, 0, *(graph->numVertices) * sizeof(bool));

  uint32 *d_edges;
  GPUAssert(cudaMalloc(&d_edges, graph->numEdges * sizeof(*d_edges)));

  std::cout << "Starting Traversals" << std::endl;
  for (int test = 0; test < nRuns; test++) {

    graph->ResetFrontierNValues();

    totalProcess.startRecord();

#ifdef ZC
    uint32 numBlocks =
        (((*(graph->numVertices)) * WARP_SIZE + THREADS_PER_BLOCK) /
         THREADS_PER_BLOCK);
    dim3 gridDim(THREADS_PER_BLOCK,
                 (numBlocks + THREADS_PER_BLOCK) / THREADS_PER_BLOCK);

    CalculateCostNSplitFrontiers15<<<gridDim, blockDim, 0, streams[0]>>>(
        graph->numVertices, graph->d_values, graph->d_frontier, graph->h_edges,
        graph->d_offsets);
#else

    //   cudaMemcpyAsync(d_edges, graph->h_edges, graph->numEdges *
    //   sizeof(*d_edges),
    //                   cudaMemcpyHostToDevice, streams[0]);

    //   uint32 numBlocks =
    //       (((*(graph->numVertices)) * WARP_SIZE + THREADS_PER_BLOCK) /
    //        THREADS_PER_BLOCK);
    //   dim3 gridDim(THREADS_PER_BLOCK,
    //                (numBlocks + THREADS_PER_BLOCK) / THREADS_PER_BLOCK);

    //   CalculateCostNSplitFrontiers16<<<gridDim, blockDim, 0, streams[0]>>>(
    //       graph->numVertices, graph->d_values, graph->d_frontier, d_edges,
    //       graph->d_offsets);

    uint32 *d_partitionList0;
    uint32 *d_partitionList1;
    uint32 *d_partitionList2;
    uint32 *d_partitionList3;
    uint32 *d_partitionList4;
    uint32 *d_partitionList5;
    uint32 *d_partitionList6;
    uint32 *d_partitionList7;
    uint32 *d_partitionList8;
    uint32 *d_partitionList9;
    uint32 *d_partitionList10;
    uint32 *d_partitionList11;
    uint32 *d_partitionList12;
    uint32 *d_partitionList13;
    uint32 *d_partitionList14;
    uint32 *d_partitionList15;
    GPUAssert(cudaMalloc(&d_partitionList0, sizeof(*d_partitionList0)));
    GPUAssert(cudaMalloc(&d_partitionList1, sizeof(*d_partitionList0)));
    GPUAssert(cudaMalloc(&d_partitionList2, sizeof(*d_partitionList0)));
    GPUAssert(cudaMalloc(&d_partitionList3, sizeof(*d_partitionList0)));
    GPUAssert(cudaMalloc(&d_partitionList4, sizeof(*d_partitionList0)));
    GPUAssert(cudaMalloc(&d_partitionList5, sizeof(*d_partitionList0)));
    GPUAssert(cudaMalloc(&d_partitionList6, sizeof(*d_partitionList0)));
    GPUAssert(cudaMalloc(&d_partitionList7, sizeof(*d_partitionList0)));
    GPUAssert(cudaMalloc(&d_partitionList8, sizeof(*d_partitionList0)));
    GPUAssert(cudaMalloc(&d_partitionList9, sizeof(*d_partitionList0)));
    GPUAssert(cudaMalloc(&d_partitionList10, sizeof(*d_partitionList0)));
    GPUAssert(cudaMalloc(&d_partitionList11, sizeof(*d_partitionList0)));
    GPUAssert(cudaMalloc(&d_partitionList12, sizeof(*d_partitionList0)));
    GPUAssert(cudaMalloc(&d_partitionList13, sizeof(*d_partitionList0)));
    GPUAssert(cudaMalloc(&d_partitionList14, sizeof(*d_partitionList0)));
    GPUAssert(cudaMalloc(&d_partitionList15, sizeof(*d_partitionList0)));

    uint32 *d_filterEdges0;
    uint32 *d_filterEdges1;
    uint32 *d_filterEdges2;
    uint32 *d_filterEdges3;
    uint32 *d_filterEdges4;
    uint32 *d_filterEdges5;
    uint32 *d_filterEdges6;
    uint32 *d_filterEdges7;
    uint32 *d_filterEdges8;
    uint32 *d_filterEdges9;
    uint32 *d_filterEdges10;
    uint32 *d_filterEdges11;
    uint32 *d_filterEdges12;
    uint32 *d_filterEdges13;
    uint32 *d_filterEdges14;
    uint32 *d_filterEdges15;

    GPUAssert(cudaMalloc(&d_filterEdges0,
                         EDGES_IN_PARTITION * sizeof(*d_filterEdges0)));
    GPUAssert(cudaMalloc(&d_filterEdges1,
                         EDGES_IN_PARTITION * sizeof(*d_filterEdges1)));
    GPUAssert(cudaMalloc(&d_filterEdges2,
                         EDGES_IN_PARTITION * sizeof(*d_filterEdges2)));
    GPUAssert(cudaMalloc(&d_filterEdges3,
                         EDGES_IN_PARTITION * sizeof(*d_filterEdges3)));
    GPUAssert(cudaMalloc(&d_filterEdges4,
                         EDGES_IN_PARTITION * sizeof(*d_filterEdges4)));
    GPUAssert(cudaMalloc(&d_filterEdges5,
                         EDGES_IN_PARTITION * sizeof(*d_filterEdges5)));
    GPUAssert(cudaMalloc(&d_filterEdges6,
                         EDGES_IN_PARTITION * sizeof(*d_filterEdges6)));
    GPUAssert(cudaMalloc(&d_filterEdges7,
                         EDGES_IN_PARTITION * sizeof(*d_filterEdges7)));
    GPUAssert(cudaMalloc(&d_filterEdges8,
                         EDGES_IN_PARTITION * sizeof(*d_filterEdges8)));
    GPUAssert(cudaMalloc(&d_filterEdges9,
                         EDGES_IN_PARTITION * sizeof(*d_filterEdges9)));
    GPUAssert(cudaMalloc(&d_filterEdges10,
                         EDGES_IN_PARTITION * sizeof(*d_filterEdges10)));
    GPUAssert(cudaMalloc(&d_filterEdges11,
                         EDGES_IN_PARTITION * sizeof(*d_filterEdges11)));
    GPUAssert(cudaMalloc(&d_filterEdges12,
                         EDGES_IN_PARTITION * sizeof(*d_filterEdges12)));
    GPUAssert(cudaMalloc(&d_filterEdges13,
                         EDGES_IN_PARTITION * sizeof(*d_filterEdges13)));
    GPUAssert(cudaMalloc(&d_filterEdges14,
                         EDGES_IN_PARTITION * sizeof(*d_filterEdges14)));
    GPUAssert(cudaMalloc(&d_filterEdges15,
                         EDGES_IN_PARTITION * sizeof(*d_filterEdges15)));

    for (uint32 partition = 0; partition < *graph->numPartitions; partition++) {

      // Partition edge start
      uint32 start = graph->h_offsets[graph->h_partitionsOffsets[partition]];
      // Partition edge end
      uint32 end = graph->h_offsets[graph->h_partitionsOffsets[partition + 1]];

      uint32 partitionSize = end - start;

      uint32 stream = partition % N_FILTER_STREAMS2;

      graph->h_partitionList[stream] = partition;

      switch (stream) {
      case 0:
        cudaMemcpyAsync(d_filterEdges0, graph->h_edges + start,
                        partitionSize * sizeof(*graph->h_edges),
                        cudaMemcpyHostToDevice, streams[stream]);

        cudaMemcpyAsync(d_partitionList0, graph->h_partitionList + stream,
                        sizeof(*graph->h_partitionList), cudaMemcpyHostToDevice,
                        streams[stream]);
        CalculateCostNSplitFrontiers11<<<staticGrid, blockDim, 0,
                                         streams[stream]>>>(
            d_partitionList0, graph->d_partitionsOffsets, graph->d_values,
            graph->d_frontier, d_filterEdges0, graph->d_offsets);
        break;
      case 1:
        cudaMemcpyAsync(d_filterEdges1, graph->h_edges + start,
                        partitionSize * sizeof(*graph->h_edges),
                        cudaMemcpyHostToDevice, streams[stream]);

        cudaMemcpyAsync(d_partitionList1, graph->h_partitionList + stream,
                        sizeof(*graph->h_partitionList), cudaMemcpyHostToDevice,
                        streams[stream]);
        CalculateCostNSplitFrontiers11<<<staticGrid, blockDim, 0,
                                         streams[stream]>>>(
            d_partitionList1, graph->d_partitionsOffsets, graph->d_values,
            graph->d_frontier, d_filterEdges1, graph->d_offsets);
        break;
      case 2:
        cudaMemcpyAsync(d_filterEdges2, graph->h_edges + start,
                        partitionSize * sizeof(*graph->h_edges),
                        cudaMemcpyHostToDevice, streams[stream]);

        cudaMemcpyAsync(d_partitionList2, graph->h_partitionList + stream,
                        sizeof(*graph->h_partitionList), cudaMemcpyHostToDevice,
                        streams[stream]);
        CalculateCostNSplitFrontiers11<<<staticGrid, blockDim, 0,
                                         streams[stream]>>>(
            d_partitionList2, graph->d_partitionsOffsets, graph->d_values,
            graph->d_frontier, d_filterEdges2, graph->d_offsets);
        break;
      case 3:
        cudaMemcpyAsync(d_filterEdges3, graph->h_edges + start,
                        partitionSize * sizeof(*graph->h_edges),
                        cudaMemcpyHostToDevice, streams[stream]);

        cudaMemcpyAsync(d_partitionList3, graph->h_partitionList + stream,
                        sizeof(*graph->h_partitionList), cudaMemcpyHostToDevice,
                        streams[stream]);
        CalculateCostNSplitFrontiers11<<<staticGrid, blockDim, 0,
                                         streams[stream]>>>(
            d_partitionList3, graph->d_partitionsOffsets, graph->d_values,
            graph->d_frontier, d_filterEdges3, graph->d_offsets);
        break;
      case 4:
        cudaMemcpyAsync(d_filterEdges4, graph->h_edges + start,
                        partitionSize * sizeof(*graph->h_edges),
                        cudaMemcpyHostToDevice, streams[stream]);

        cudaMemcpyAsync(d_partitionList4, graph->h_partitionList + stream,
                        sizeof(*graph->h_partitionList), cudaMemcpyHostToDevice,
                        streams[stream]);
        CalculateCostNSplitFrontiers11<<<staticGrid, blockDim, 0,
                                         streams[stream]>>>(
            d_partitionList4, graph->d_partitionsOffsets, graph->d_values,
            graph->d_frontier, d_filterEdges4, graph->d_offsets);
        break;
      case 5:
        cudaMemcpyAsync(d_filterEdges5, graph->h_edges + start,
                        partitionSize * sizeof(*graph->h_edges),
                        cudaMemcpyHostToDevice, streams[stream]);

        cudaMemcpyAsync(d_partitionList5, graph->h_partitionList + stream,
                        sizeof(*graph->h_partitionList), cudaMemcpyHostToDevice,
                        streams[stream]);
        CalculateCostNSplitFrontiers11<<<staticGrid, blockDim, 0,
                                         streams[stream]>>>(
            d_partitionList5, graph->d_partitionsOffsets, graph->d_values,
            graph->d_frontier, d_filterEdges5, graph->d_offsets);
        break;
      case 6:
        cudaMemcpyAsync(d_filterEdges6, graph->h_edges + start,
                        partitionSize * sizeof(*graph->h_edges),
                        cudaMemcpyHostToDevice, streams[stream]);

        cudaMemcpyAsync(d_partitionList6, graph->h_partitionList + stream,
                        sizeof(*graph->h_partitionList), cudaMemcpyHostToDevice,
                        streams[stream]);
        CalculateCostNSplitFrontiers11<<<staticGrid, blockDim, 0,
                                         streams[stream]>>>(
            d_partitionList6, graph->d_partitionsOffsets, graph->d_values,
            graph->d_frontier, d_filterEdges6, graph->d_offsets);
        break;
      case 7:
        cudaMemcpyAsync(d_filterEdges7, graph->h_edges + start,
                        partitionSize * sizeof(*graph->h_edges),
                        cudaMemcpyHostToDevice, streams[stream]);

        cudaMemcpyAsync(d_partitionList7, graph->h_partitionList + stream,
                        sizeof(*graph->h_partitionList), cudaMemcpyHostToDevice,
                        streams[stream]);
        CalculateCostNSplitFrontiers11<<<staticGrid, blockDim, 0,
                                         streams[stream]>>>(
            d_partitionList7, graph->d_partitionsOffsets, graph->d_values,
            graph->d_frontier, d_filterEdges7, graph->d_offsets);
        break;
      case 8:
        cudaMemcpyAsync(d_filterEdges8, graph->h_edges + start,
                        partitionSize * sizeof(*graph->h_edges),
                        cudaMemcpyHostToDevice, streams[stream]);

        cudaMemcpyAsync(d_partitionList8, graph->h_partitionList + stream,
                        sizeof(*graph->h_partitionList), cudaMemcpyHostToDevice,
                        streams[stream]);
        CalculateCostNSplitFrontiers11<<<staticGrid, blockDim, 0,
                                         streams[stream]>>>(
            d_partitionList8, graph->d_partitionsOffsets, graph->d_values,
            graph->d_frontier, d_filterEdges8, graph->d_offsets);
        break;
      case 9:
        cudaMemcpyAsync(d_filterEdges9, graph->h_edges + start,
                        partitionSize * sizeof(*graph->h_edges),
                        cudaMemcpyHostToDevice, streams[stream]);

        cudaMemcpyAsync(d_partitionList9, graph->h_partitionList + stream,
                        sizeof(*graph->h_partitionList), cudaMemcpyHostToDevice,
                        streams[stream]);
        CalculateCostNSplitFrontiers11<<<staticGrid, blockDim, 0,
                                         streams[stream]>>>(
            d_partitionList9, graph->d_partitionsOffsets, graph->d_values,
            graph->d_frontier, d_filterEdges9, graph->d_offsets);
        break;
      case 10:
        cudaMemcpyAsync(d_filterEdges10, graph->h_edges + start,
                        partitionSize * sizeof(*graph->h_edges),
                        cudaMemcpyHostToDevice, streams[stream]);

        cudaMemcpyAsync(d_partitionList10, graph->h_partitionList + stream,
                        sizeof(*graph->h_partitionList), cudaMemcpyHostToDevice,
                        streams[stream]);
        CalculateCostNSplitFrontiers11<<<staticGrid, blockDim, 0,
                                         streams[stream]>>>(
            d_partitionList10, graph->d_partitionsOffsets, graph->d_values,
            graph->d_frontier, d_filterEdges10, graph->d_offsets);
        break;
      case 11:
        cudaMemcpyAsync(d_filterEdges11, graph->h_edges + start,
                        partitionSize * sizeof(*graph->h_edges),
                        cudaMemcpyHostToDevice, streams[stream]);

        cudaMemcpyAsync(d_partitionList11, graph->h_partitionList + stream,
                        sizeof(*graph->h_partitionList), cudaMemcpyHostToDevice,
                        streams[stream]);
        CalculateCostNSplitFrontiers11<<<staticGrid, blockDim, 0,
                                         streams[stream]>>>(
            d_partitionList11, graph->d_partitionsOffsets, graph->d_values,
            graph->d_frontier, d_filterEdges11, graph->d_offsets);
        break;
      case 12:
        cudaMemcpyAsync(d_filterEdges12, graph->h_edges + start,
                        partitionSize * sizeof(*graph->h_edges),
                        cudaMemcpyHostToDevice, streams[stream]);

        cudaMemcpyAsync(d_partitionList12, graph->h_partitionList + stream,
                        sizeof(*graph->h_partitionList), cudaMemcpyHostToDevice,
                        streams[stream]);
        CalculateCostNSplitFrontiers11<<<staticGrid, blockDim, 0,
                                         streams[stream]>>>(
            d_partitionList12, graph->d_partitionsOffsets, graph->d_values,
            graph->d_frontier, d_filterEdges12, graph->d_offsets);
        break;
      case 13:
        cudaMemcpyAsync(d_filterEdges13, graph->h_edges + start,
                        partitionSize * sizeof(*graph->h_edges),
                        cudaMemcpyHostToDevice, streams[stream]);

        cudaMemcpyAsync(d_partitionList13, graph->h_partitionList + stream,
                        sizeof(*graph->h_partitionList), cudaMemcpyHostToDevice,
                        streams[stream]);
        CalculateCostNSplitFrontiers11<<<staticGrid, blockDim, 0,
                                         streams[stream]>>>(
            d_partitionList13, graph->d_partitionsOffsets, graph->d_values,
            graph->d_frontier, d_filterEdges13, graph->d_offsets);
        break;
      case 14:
        cudaMemcpyAsync(d_filterEdges14, graph->h_edges + start,
                        partitionSize * sizeof(*graph->h_edges),
                        cudaMemcpyHostToDevice, streams[stream]);

        cudaMemcpyAsync(d_partitionList14, graph->h_partitionList + stream,
                        sizeof(*graph->h_partitionList), cudaMemcpyHostToDevice,
                        streams[stream]);
        CalculateCostNSplitFrontiers11<<<staticGrid, blockDim, 0,
                                         streams[stream]>>>(
            d_partitionList14, graph->d_partitionsOffsets, graph->d_values,
            graph->d_frontier, d_filterEdges14, graph->d_offsets);
        break;
      case 15:
        cudaMemcpyAsync(d_filterEdges15, graph->h_edges + start,
                        partitionSize * sizeof(*graph->h_edges),
                        cudaMemcpyHostToDevice, streams[stream]);

        cudaMemcpyAsync(d_partitionList15, graph->h_partitionList + stream,
                        sizeof(*graph->h_partitionList), cudaMemcpyHostToDevice,
                        streams[stream]);
        CalculateCostNSplitFrontiers11<<<staticGrid, blockDim, 0,
                                         streams[stream]>>>(
            d_partitionList15, graph->d_partitionsOffsets, graph->d_values,
            graph->d_frontier, d_filterEdges15, graph->d_offsets);
        break;
      }
      // GPUAssert(cudaPeekAtLastError());

      //    cudaMemcpyAsync(graph->d_partitionList + stream,
      //                      graph->h_partitionList + stream,
      //                      sizeof(*graph->h_partitionList),
      //                      cudaMemcpyHostToDevice, streams[stream]);
      //
      // Touch the data afte using all 16 streams (256MB)
      // if (stream == 15) {
      //  uint32 numBlocks =
      //      (((*(graph->numVertices)) * WARP_SIZE + THREADS_PER_BLOCK) /
      //       THREADS_PER_BLOCK);
      //  dim3 gridDim(THREADS_PER_BLOCK,
      //               (numBlocks + THREADS_PER_BLOCK) / THREADS_PER_BLOCK);

      //  CalculateCostNSplitFrontiers14<<<gridDim, blockDim, 0,
      //                                   streams[stream]>>>(
      //      graph->d_partitionList, graph->d_partitionsOffsets,
      //      graph->d_values, graph->d_frontier, graph->d_filterEdges,
      //      graph->d_offsets);

      //  //  GPUAssert(cudaPeekAtLastError());
      //}
    }

#endif
  }
  GPUAssert(cudaDeviceSynchronize());

  totalProcess.endRecord();
  totalProcess.print();

  return;
}

//      if (*graph->frontierSize > 200000) {
//
//        // cudaDeviceSynchronize();
//        CalculateCostNSplitFrontiers<uint32>
//            <<<staticGrid, blockDim, 0, partitionStream>>>(
//                graph->numPartitions, graph->d_partitionsOffsets,
//                graph->d_offsets, graph->d_zerocopyPartitionCost,
//                graph->d_demandFrontier, graph->d_filterFrontier);
//
//        CalculateCostNSplitFrontiers2<uint32>
//            <<<staticGrid, blockDim, 0, partitionStream>>>(
//                graph->numPartitions, graph->d_partitionsOffsets,
//                graph->d_offsets, graph->d_zerocopyPartitionCost,
//                graph->d_demandFrontier, graph->d_filterFrontier);
//
//        CalculateCostNSplitFrontiers3<uint32>
//            <<<staticGrid, blockDim, 0, partitionStream>>>(
//                graph->numPartitions, graph->d_partitionsOffsets,
//                graph->d_offsets, graph->d_zerocopyPartitionCost,
//                graph->d_demandFrontier, graph->d_filterFrontier);
//
//        cudaMemcpyAsync(graph->h_partitionCost,
//        graph->d_zerocopyPartitionCost,
//                        *(graph->numPartitions) *
//                            sizeof(*graph->h_partitionCost),
//                        cudaMemcpyDeviceToHost, partitionStream);
//
//        cudaMemsetAsync(graph->d_zerocopyPartitionCost, 0,
//                        *(graph->numPartitions) *
//                            sizeof(*graph->h_partitionCost),
//                        partitionStream);
//
//        cudaStreamSynchronize(partitionStream);
//      }
