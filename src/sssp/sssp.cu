#include "sssp.cuh"
#include "sssp_kernels.cuh"
#include <iostream>
#include <ostream>

#include <numa.h>
#include <queue>
#include <vector>

void SSSP32(std::string filePath, uint32 srcVertex, uint32 nRuns,
            uint32 nNeighborGPUs, std::unordered_map<int, int> affinityMap) {

  numa_run_on_node(0);
  ALGORITHM_TYPE algo = SSSP;
  CSR<uint32> *graph = new CSR<uint32>;
  graph->ReadInputFile2(filePath, algo, srcVertex, nNeighborGPUs, affinityMap);

  cudaStream_t staticStream, demandStream, frontierStream;

  GPUAssert(cudaStreamCreate(&frontierStream));
  GPUAssert(cudaStreamCreate(&staticStream));
  GPUAssert(cudaStreamCreate(&demandStream));
  uint32 nGPUs = nNeighborGPUs + 1;

  std::vector<std::array<cudaStream_t, N_FILTER_STREAMS>> neighborMemCpyStreams(
      nNeighborGPUs);

  std::vector<std::array<cudaStream_t, N_FILTER_STREAMS>>
      neighborComputeStreams(nNeighborGPUs);

  for (int i = 0; i < nNeighborGPUs; ++i) {
    cudaSetDevice(i + 1);
    for (int j = 0; j < N_FILTER_STREAMS; ++j)
      GPUAssert(cudaStreamCreate(&neighborMemCpyStreams[i][j]));
  }

  cudaSetDevice(0);

  for (int i = 0; i < nNeighborGPUs; ++i) {
    for (int j = 0; j < N_FILTER_STREAMS; ++j)
      GPUAssert(cudaStreamCreate(&neighborComputeStreams[i][j]));
  }

  cudaStream_t streams[N_TARGET_FILTER_STREAMS];

  for (uint32 i = 0; i < N_TARGET_FILTER_STREAMS; i++)
    GPUAssert(cudaStreamCreate(&streams[i]));

  graph->InitData();

  int device = 0; // Selected device
  uint32 k = 4;

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);
  uint32 numSMs = prop.multiProcessorCount;

  dim3 staticGrid = dim3(k * numSMs, 1, 1);
  dim3 blockDim(THREADS_PER_BLOCK, 1, 1); // (x,y,z) = (THREADS_PER_BLOCK, 1, 1)

  auto asyncFrontierPolicy = thrust::cuda::par_nosync.on(frontierStream);
  auto asyncStaticPolicy = thrust::cuda::par_nosync.on(staticStream);
  auto asyncDemandPolicy = thrust::cuda::par_nosync.on(demandStream);
  // auto syncPolicy  = thrust::cuda::par.on(staticStream);

  auto syncFrontierPolicy = thrust::cuda::par.on(frontierStream);
  auto syncStaticPolicy = thrust::cuda::par.on(staticStream);
  auto syncDemandPolicy = thrust::cuda::par.on(demandStream);

  uint64 totalNumFilterPartitions = 0;
  std::cout << "Starting Traversals" << std::endl;
  for (int test = 0; test < nRuns; test++) {

    graph->ResetFrontierNValues();

    *(graph->frontierSize) = thrust::reduce(
        graph->thrustFrontier, graph->thrustFrontier + *(graph->numVertices), 0,
        thrust::plus<uint32>());

    Timer timer("Execution time: ");
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

      if (*graph->frontierSize > 10 * graph->avgVertPerPart) {
        CalculateActiveEdgesPerPartition<uint32>
            <<<staticGrid, blockDim, 0, demandStream>>>(
                graph->numPartitions, graph->d_partitionsOffsets,
                graph->d_offsets, graph->d_partitionCost,
                graph->d_demandFrontier, graph->d_filterFrontier);

        CalculateActiveEdgesRatio<uint32>
            <<<staticGrid, blockDim, 0, demandStream>>>(
                graph->numPartitions, graph->d_partitionsOffsets,
                graph->d_offsets, graph->d_partitionCost,
                graph->d_demandFrontier, graph->d_filterFrontier);

        SplitZeroCopyNFilterFrontiers<uint32>
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

        cudaStreamSynchronize(frontierStream);

        SSSP32_Demand_Kernel<<<gridDim, blockDim, 0, demandStream>>>(
            graph->demandSize, graph->d_demandList, graph->d_values,
            graph->d_frontier, graph->h_edges2[graph->GPUAffinityMap[0]],
            graph->h_weights2[graph->GPUAffinityMap[0]], graph->d_offsets);
      }

      if (*graph->frontierSize > 10 * graph->avgVertPerPart) {
        uint32 numPartitionsOnTarget = 0;
        uint32 numPartitionsOnNeighbors = 0;

        std::queue<uint32> targetGPUQueue;
        std::vector<std::queue<uint32>> neighborGPUQueues(nNeighborGPUs);

        std::vector<uint32> partitionList;

        cudaStreamSynchronize(streams[0]);

        for (uint32 partition = 0; partition < *graph->numPartitions;
             partition++) {

          if (graph->h_partitionCost[partition] <= graph->h_filterThreshold)
            continue;

          partitionList.push_back(partition);
        }

        cudaStreamSynchronize(frontierStream);

        totalNumFilterPartitions += partitionList.size();

        for (uint32 index = 0; index < partitionList.size(); index++) {

          uint32 partition = partitionList[index];

          // Partition edge start
          uint32 start =
              graph->h_offsets[graph->h_partitionsOffsets[partition]];

          uint32 partitionSize =
              graph->h_offsets[graph->h_partitionsOffsets[partition + 1]] -
              start;

          uint32 stream = (index / nGPUs) % N_TARGET_FILTER_STREAMS;

          graph->h_partitionList[stream] = partition;

          cudaStreamSynchronize(streams[stream]);

          // cudaDeviceSynchronize();
          cudaMemcpyAsync(graph->d_filterEdges[stream],
                          &graph->h_edges2[graph->GPUAffinityMap[0]][start],
                          partitionSize * sizeof(*graph->h_edges),
                          cudaMemcpyHostToDevice, streams[stream]);

          cudaMemcpyAsync(graph->d_filterWeights[stream],
                          &graph->h_weights2[graph->GPUAffinityMap[0]][start],
                          partitionSize * sizeof(*graph->h_weights),
                          cudaMemcpyHostToDevice, streams[stream]);

          cudaMemcpyAsync(&graph->d_partitionList[stream],
                          &graph->h_partitionList[stream],
                          sizeof(*graph->h_partitionList),
                          cudaMemcpyHostToDevice, streams[stream]);

          //  cudaDeviceSynchronize();

          targetGPUQueue.push(stream);

          for (uint32 gpu = 0; gpu < neighborGPUQueues.size(); gpu++) {
            if (index + 1 >= partitionList.size())
              break;

            index++;
            partition = partitionList[index];

            // Partition edge start
            uint32 neighborStart =
                graph->h_offsets[graph->h_partitionsOffsets[partition]];

            uint32 neighborPartitionSize =
                graph->h_offsets[graph->h_partitionsOffsets[partition + 1]] -
                neighborStart;

            uint32 neighborStream = (index / nGPUs) % N_FILTER_STREAMS;

            graph->h_nPartList[gpu][neighborStream] = partition;

            // Sync compute stream on device 0
            cudaStreamSynchronize(neighborComputeStreams[gpu][neighborStream]);
            // cudaDeviceSynchronize();
            cudaSetDevice(gpu + 1);

            //   cudaDeviceSynchronize();

            cudaMemcpyAsync(graph->d_nFilterEdges[gpu][neighborStream],
                            graph->h_edges2[graph->GPUAffinityMap[0]] +
                                neighborStart,
                            neighborPartitionSize * sizeof(*graph->h_edges),
                            cudaMemcpyHostToDevice,
                            neighborMemCpyStreams[gpu][neighborStream]);

            cudaMemcpyAsync(graph->d_nFilterWeights[gpu][neighborStream],
                            graph->h_weights2[graph->GPUAffinityMap[0]] +
                                neighborStart,
                            neighborPartitionSize * sizeof(*graph->h_weights),
                            cudaMemcpyHostToDevice,
                            neighborMemCpyStreams[gpu][neighborStream]);

            // We can prob allocate this data in the other numa node too
            cudaMemcpyAsync(&graph->d_nPartList[gpu][neighborStream],
                            graph->h_nPartList[gpu] + neighborStream,
                            sizeof(**graph->h_nPartList),
                            cudaMemcpyHostToDevice,
                            neighborMemCpyStreams[gpu][neighborStream]);

            //  cudaDeviceSynchronize();

            neighborGPUQueues[gpu].push(neighborStream);

            cudaSetDevice(0);
          }

          // while (!targetGPUQueue.empty())
          {
            uint32 tStream = targetGPUQueue.front();

            cudaError_t streamStatus = cudaStreamQuery(streams[tStream]);

            if (streamStatus == cudaErrorNotReady) {
              if (targetGPUQueue.size() < N_TARGET_FILTER_STREAMS)
                continue;
              else
                cudaStreamSynchronize(streams[tStream]);
            }

            targetGPUQueue.pop();

            numPartitionsOnTarget++;
            //   cudaDeviceSynchronize();
            SSSP32_Filter_Kernel<<<staticGrid, blockDim, 0, streams[tStream]>>>(
                &graph->d_partitionList[tStream], graph->d_partitionsOffsets,
                graph->d_values, graph->d_frontier,
                graph->d_filterEdges[tStream], graph->d_offsets,
                graph->d_filterFrontier, graph->d_filterWeights[tStream]);
            //       cudaDeviceSynchronize();

            uint32 processedPartition = graph->h_partitionList[tStream];

            uint32 partitionStart =
                graph
                    ->h_offsets[graph->h_partitionsOffsets[processedPartition]];

            uint32 partitionEnd =
                graph->h_offsets[graph->h_partitionsOffsets[processedPartition +
                                                            1]];

            uint32 processedPartitionSize = partitionEnd - partitionStart;

            if (partitionEnd <= graph->numStaticEdges) {

              // cudaStreamSynchronize(staticStreams[processedPartition]);
              cudaMemcpyAsync(&graph->d_staticEdges[partitionStart],
                              graph->d_filterEdges[tStream],
                              processedPartitionSize * sizeof(*graph->h_edges),
                              cudaMemcpyDeviceToDevice, streams[tStream]);

              cudaMemcpyAsync(&graph->d_staticWeights[partitionStart],
                              graph->d_filterWeights[tStream],
                              processedPartitionSize * sizeof(*graph->h_edges),
                              cudaMemcpyDeviceToDevice, streams[tStream]);

              cudaMemsetAsync(
                  &graph->d_inStatic
                       [graph->h_partitionsOffsets[processedPartition]],
                  1,
                  (graph->h_partitionsOffsets[processedPartition + 1] -
                   graph->h_partitionsOffsets[processedPartition]) *
                      sizeof(*graph->d_inStatic),
                  streams[tStream]);
              // cudaDeviceSynchronize();
            }
          }

          for (uint32 gpu = 0; gpu < neighborGPUQueues.size(); gpu++) {

            // while (!neighborGPUQueues[gpu].empty())
            {
              uint32 nStream = neighborGPUQueues[gpu].front();

              cudaSetDevice(gpu + 1);
              cudaError_t streamStatus =
                  cudaStreamQuery(neighborMemCpyStreams[gpu][nStream]);

              if (streamStatus == cudaErrorNotReady) {
                if (neighborGPUQueues[gpu].size() < N_FILTER_STREAMS) {
                  cudaSetDevice(0);
                  continue;
                } else
                  cudaStreamSynchronize(neighborMemCpyStreams[gpu][nStream]);
              }

              cudaSetDevice(0);
              neighborGPUQueues[gpu].pop();

              numPartitionsOnNeighbors++;
              uint32 processedPartition = graph->h_nPartList[gpu][nStream];

              uint32 partitionStart =
                  graph->h_offsets
                      [graph->h_partitionsOffsets[processedPartition]];

              uint32 partitionEnd =
                  graph->h_offsets
                      [graph->h_partitionsOffsets[processedPartition + 1]];

              uint32 processedPartitionSize = partitionEnd - partitionStart;

              if (partitionEnd <= graph->numStaticEdges) {

                // Aqui
                // cudaSetDevice(gpu + 1);
                cudaMemcpyAsync(&graph->d_staticEdges[partitionStart],
                                graph->d_nFilterEdges[gpu][nStream],
                                processedPartitionSize *
                                    sizeof(*graph->h_edges),
                                cudaMemcpyDeviceToDevice,
                                neighborComputeStreams[gpu][nStream]);

                // cudaSetDevice(0);

                cudaMemsetAsync(
                    &graph->d_inStatic
                         [graph->h_partitionsOffsets[processedPartition]],
                    1,
                    (graph->h_partitionsOffsets[processedPartition + 1] -
                     graph->h_partitionsOffsets[processedPartition]) *
                        sizeof(*graph->d_inStatic),
                    neighborComputeStreams[gpu][nStream]);

                // cudaDeviceSynchronize();
                SSSP32_Static_Filter_Kernel<<<
                    staticGrid, blockDim, 0,
                    neighborComputeStreams[gpu][nStream]>>>(
                    &graph->d_nPartList[gpu][nStream],
                    graph->d_partitionsOffsets, graph->d_values,
                    graph->d_frontier, graph->d_staticEdges, graph->d_offsets,
                    graph->d_filterFrontier, graph->d_staticWeights);
              } else {
                SSSP32_NeighborFilter_Kernel<<<
                    staticGrid, blockDim, 0,
                    neighborComputeStreams[gpu][nStream]>>>(
                    &graph->d_nPartList[gpu][nStream],
                    graph->d_partitionsOffsets, graph->d_values,
                    graph->d_frontier, graph->d_nFilterEdges[gpu][nStream],
                    graph->d_offsets, graph->d_filterFrontier,
                    graph->d_nFilterWeights[gpu][nStream]);
              }
            }
          }
        }

        while (!targetGPUQueue.empty()) {

          uint32 tStream = targetGPUQueue.front();

          cudaStreamSynchronize(streams[tStream]);
          targetGPUQueue.pop();

          numPartitionsOnTarget++;
          SSSP32_Filter_Kernel<<<staticGrid, blockDim, 0, streams[tStream]>>>(
              &graph->d_partitionList[tStream], graph->d_partitionsOffsets,
              graph->d_values, graph->d_frontier, graph->d_filterEdges[tStream],
              graph->d_offsets, graph->d_filterFrontier,
              graph->d_filterWeights[tStream]);
          //   cudaDeviceSynchronize();

          uint32 processedPartition = graph->h_partitionList[tStream];

          uint32 partitionStart =
              graph->h_offsets[graph->h_partitionsOffsets[processedPartition]];

          uint32 partitionEnd =
              graph->h_offsets[graph->h_partitionsOffsets[processedPartition +
                                                          1]];

          uint32 processedPartitionSize = partitionEnd - partitionStart;

          if (partitionEnd <= graph->numStaticEdges) {

            // cudaStreamSynchronize(staticStreams[processedPartition]);
            cudaMemcpyAsync(&graph->d_staticEdges[partitionStart],
                            graph->d_filterEdges[tStream],
                            processedPartitionSize * sizeof(*graph->h_edges),
                            cudaMemcpyDeviceToDevice, streams[tStream]);

            cudaMemcpyAsync(&graph->d_staticWeights[partitionStart],
                            graph->d_filterWeights[tStream],
                            processedPartitionSize * sizeof(*graph->h_edges),
                            cudaMemcpyDeviceToDevice, streams[tStream]);

            cudaMemsetAsync(
                &graph->d_inStatic
                     [graph->h_partitionsOffsets[processedPartition]],
                1,
                (graph->h_partitionsOffsets[processedPartition + 1] -
                 graph->h_partitionsOffsets[processedPartition]) *
                    sizeof(*graph->d_inStatic),
                streams[tStream]);
            // cudaDeviceSynchronize();
          }
        }

        for (uint32 gpu = 0; gpu < neighborGPUQueues.size(); gpu++) {

          while (!neighborGPUQueues[gpu].empty()) {
            uint32 nStream = neighborGPUQueues[gpu].front();

            cudaSetDevice(gpu + 1);
            cudaError_t streamStatus =
                cudaStreamQuery(neighborMemCpyStreams[gpu][nStream]);

            if (streamStatus == cudaErrorNotReady) {
              if (neighborGPUQueues[gpu].size() < N_FILTER_STREAMS) {
                cudaSetDevice(0);
                continue;
              } else
                cudaStreamSynchronize(neighborMemCpyStreams[gpu][nStream]);
            }

            cudaSetDevice(0);
            neighborGPUQueues[gpu].pop();

            numPartitionsOnNeighbors++;

            uint32 processedPartition = graph->h_nPartList[gpu][nStream];

            uint32 partitionStart =
                graph
                    ->h_offsets[graph->h_partitionsOffsets[processedPartition]];

            uint32 partitionEnd =
                graph->h_offsets[graph->h_partitionsOffsets[processedPartition +
                                                            1]];

            uint32 processedPartitionSize = partitionEnd - partitionStart;

            if (partitionEnd <= graph->numStaticEdges) {

              // Aqui
              // cudaSetDevice(gpu + 1);
              cudaMemcpyAsync(&graph->d_staticEdges[partitionStart],
                              graph->d_nFilterEdges[gpu][nStream],
                              processedPartitionSize * sizeof(*graph->h_edges),
                              cudaMemcpyDeviceToDevice,
                              neighborComputeStreams[gpu][nStream]);

              // cudaSetDevice(0);

              cudaMemsetAsync(
                  &graph->d_inStatic
                       [graph->h_partitionsOffsets[processedPartition]],
                  1,
                  (graph->h_partitionsOffsets[processedPartition + 1] -
                   graph->h_partitionsOffsets[processedPartition]) *
                      sizeof(*graph->d_inStatic),
                  neighborComputeStreams[gpu][nStream]);

              // cudaDeviceSynchronize();
              SSSP32_Static_Filter_Kernel<<<
                  staticGrid, blockDim, 0,
                  neighborComputeStreams[gpu][nStream]>>>(
                  &graph->d_nPartList[gpu][nStream], graph->d_partitionsOffsets,
                  graph->d_values, graph->d_frontier, graph->d_staticEdges,
                  graph->d_offsets, graph->d_filterFrontier,
                  graph->d_staticWeights);
            } else {
              SSSP32_NeighborFilter_Kernel<<<
                  staticGrid, blockDim, 0,
                  neighborComputeStreams[gpu][nStream]>>>(
                  &graph->d_nPartList[gpu][nStream], graph->d_partitionsOffsets,
                  graph->d_values, graph->d_frontier,
                  graph->d_nFilterEdges[gpu][nStream], graph->d_offsets,
                  graph->d_filterFrontier,
                  graph->d_nFilterWeights[gpu][nStream]);
            }

            //   cudaDeviceSynchronize();
          }
        }
        // std::cout << "Partitions processed in target GPU: "
        //           << numPartitionsOnTarget << std::endl;
        //
        // std::cout << "Partitions to be processed in neighbor GPUs: "
        //           << numPartitionsOnNeighbors << std::endl;
      }
      cudaDeviceSynchronize();

      *(graph->frontierSize) = thrust::reduce(
          graph->thrustFrontier, graph->thrustFrontier + *(graph->numVertices),
          0, thrust::plus<uint32>());
    }
  }

  const uint64 partitionSizeMB = PARTITION_SIZE_MB / (1024 * 1024); // 1024^2

  uint64 MBytes = totalNumFilterPartitions * partitionSizeMB * 2;

  // uint64 GBytes = MBytes >> 10;
  std::cout << "Total partitions in filter: " << totalNumFilterPartitions
            << std::endl;

  std::cout << "Total amount of data sent with filter: " << MBytes << " MB"
            << std::endl;

  graph->DumpValues();
  return;
}

void SSSP64(std::string filePath, uint32 srcVertex, uint32 nRuns) { return; }
