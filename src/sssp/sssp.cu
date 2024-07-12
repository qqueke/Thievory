#include "sssp.cuh"
#include <iostream>

void SSSP32(string filePath, uint32 srcVertex, double memAdvise, uint32 nRuns) {
  ALGORITHM_TYPE algo = SSSP;
  CSR<uint32> *graph = new CSR<uint32>;
  graph->ReadInputFile(filePath, algo);
  graph->InitData(srcVertex);
  // Adjust this number of blocks in x dimension to be a multiple of the number
  // of SMS and acquire better load balancing
  int device = 0; // Selected device
  uint32 k =
      4; // Multiple of SMs to choose for the grid dimension (to be adjusted)

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);
  uint32 numSMs = prop.multiProcessorCount;

  dim3 staticGrid = dim3(k * numSMs, 1, 1);
  dim3 blockDim(THREADS_PER_BLOCK, 1, 1); // (x,y,z) = (THREADS_PER_BLOCK, 1, 1)

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

  // Removing static data
  cudaMemset(graph->d_inStatic, 0, *(graph->numVertices) * sizeof(bool));

  std::cout << "Starting Traversals" << std::endl;
  for (int test = 0; test < nRuns; test++) {

    graph->ResetFrontierNValues();

    *(graph->frontierSize) = thrust::reduce(
        graph->thrustFrontier, graph->thrustFrontier + *(graph->numVertices), 0,
        thrust::plus<uint32>());

    totalProcess.startRecord();

    while (*(graph->frontierSize)) {

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

        SSSP32_Static_Kernel<<<staticGrid, blockDim, 0, staticStream>>>(
            graph->staticSize, graph->d_staticList, graph->d_offsets,
            graph->d_staticEdges, graph->d_staticWeights, graph->d_values,
            graph->d_frontier, graph->d_staticFrontier);
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

        cudaStreamSynchronize(demandStream);

        SSSP32_Demand_Kernel<<<gridDim, blockDim, 0, demandStream>>>(
            graph->demandSize, graph->d_demandList, graph->d_values,
            graph->d_frontier, graph->h_edges, graph->h_weights,
            graph->d_offsets);
      }

      GPUAssert(cudaDeviceSynchronize());

      *(graph->frontierSize) = thrust::reduce(
          graph->thrustFrontier, graph->thrustFrontier + *(graph->numVertices),
          0, thrust::plus<uint32>());
    }
  }

  totalProcess.endRecord();
  totalProcess.print();

  // We're gonna need to compare results now!!
  cudaMemcpy(graph->h_values, graph->d_values,
             *(graph->numVertices) * sizeof(uint32), cudaMemcpyDeviceToHost);

  graph->DumpValues();
  return;
}

void SSSP64(string filePath, uint32 srcVertex, double memAdvise, uint32 nRuns) {

  return;
}
