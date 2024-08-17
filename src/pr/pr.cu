#include "pr.cuh"
#include "pr_kernels.cuh"
#include <iostream>
#include <ostream>

#include <numa.h>
#include <queue>
#include <vector>

void PR32(std::string filePath, uint32 nRuns, uint32 nNeighborGPUs) {

  numa_run_on_node(0);
  ALGORITHM_TYPE algo = PR;
  CSR<uint32> *graph = new CSR<uint32>;
  graph->ReadInputFile(filePath, algo, std::string("pull"));

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

  int device = 0;
  uint32 k = 2;

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);
  uint32 numSMs = prop.multiProcessorCount;

  dim3 staticGrid = dim3(k * numSMs, 1, 1);
  dim3 blockDim(THREADS_PER_BLOCK, 1, 1); // (x,y,z) = (THREADS_PER_BLOCK, 1, 1)

  GPUAssert(cudaPeekAtLastError());

  uint64 totalNumFilterPartitions = 0;

  std::cout << "Starting Traversals" << std::endl;
  for (int test = 0; test < nRuns; test++) {

    graph->ResetFrontierNValues();

    *(graph->frontierSize) = thrust::reduce(
        graph->thrustFrontier, graph->thrustFrontier + *(graph->numVertices), 0,
        thrust::plus<uint32>());

    Timer timer("Execution time: ");
    while (*(graph->frontierSize)) {

      // std::cout << "Frontier size: " << *graph->frontierSize << std::endl;

      setStaticNDemandFrontiers<<<staticGrid, blockDim, 0, frontierStream>>>(
          graph->numVertices, graph->d_frontier, graph->d_staticFrontier,
          graph->d_demandFrontier, graph->d_inStatic);

      cudaStreamSynchronize(frontierStream);

      // Calculate the amount of active nodes in GPU memory
      *(graph->staticSize) =
          thrust::reduce(graph->thurstStaticFrontier,
                         graph->thurstStaticFrontier + *(graph->numVertices), 0,
                         thrust::plus<uint32>());

      if (*graph->frontierSize > 80 * graph->avgVertPerPart) {
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

        PR32_Static_Kernel<<<staticGrid, blockDim, 0, staticStream>>>(
            graph->staticSize, graph->d_staticList, graph->d_offsets,
            graph->d_staticEdges, graph->d_valuesPR, graph->d_degree,
            graph->d_sum, graph->d_inStatic);
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

        PR32_Demand_Kernel<<<gridDim, blockDim, 0, demandStream>>>(
            graph->demandSize, graph->d_demandList, graph->h_edges,
            graph->d_offsets, graph->d_valuesPR, graph->d_degree, graph->d_sum,
            graph->d_inStatic);
      }

      if (*graph->frontierSize > 80 * graph->avgVertPerPart) {
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
          cudaMemcpyAsync(graph->d_filterEdges[stream], &graph->h_edges[start],
                          partitionSize * sizeof(*graph->h_edges),
                          cudaMemcpyDefault, streams[stream]);

          cudaMemcpyAsync(&graph->d_partitionList[stream],
                          &graph->h_partitionList[stream],
                          sizeof(*graph->h_partitionList),
                          cudaMemcpyHostToDevice, streams[stream]);

          // GPUAssert(cudaPeekAtLastError());
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

            cudaMemcpyAsync(graph->d_nFilterEdges[gpu][neighborStream],
                            (gpu > 0) ? graph->h_edges + neighborStart
                                      : graph->h_edges + neighborStart,
                            neighborPartitionSize * sizeof(*graph->h_edges),
                            cudaMemcpyDefault,
                            neighborMemCpyStreams[gpu][neighborStream]);

            // GPUAssert(cudaPeekAtLastError());
            // We can prob allocate this data in the other numa node too
            cudaMemcpyAsync(&graph->d_nPartList[gpu][neighborStream],
                            graph->h_nPartList[gpu] + neighborStream,
                            sizeof(**graph->h_nPartList),
                            cudaMemcpyHostToDevice,
                            neighborMemCpyStreams[gpu][neighborStream]);

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
            PR32_Filter_Kernel<<<staticGrid, blockDim, 0, streams[tStream]>>>(
                &graph->d_partitionList[tStream], graph->d_partitionsOffsets,
                graph->d_values, graph->d_frontier,
                graph->d_filterEdges[tStream], graph->d_offsets,
                graph->d_filterFrontier, graph->d_valuesPR, graph->d_degree,
                graph->d_sum);

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

                PR32_Static_NeighborFilter_Kernel<<<
                    staticGrid, blockDim, 0,
                    neighborComputeStreams[gpu][nStream]>>>(
                    &graph->d_nPartList[gpu][nStream],
                    graph->d_partitionsOffsets, graph->d_values,
                    graph->d_frontier, graph->d_staticEdges, graph->d_offsets,
                    graph->d_filterFrontier, graph->d_valuesPR, graph->d_degree,
                    graph->d_sum);
              } else {
                PR32_NeighborFilter_Kernel<<<
                    staticGrid, blockDim, 0,
                    neighborComputeStreams[gpu][nStream]>>>(
                    &graph->d_nPartList[gpu][nStream],
                    graph->d_partitionsOffsets, graph->d_values,
                    graph->d_frontier, graph->d_nFilterEdges[gpu][nStream],
                    graph->d_offsets, graph->d_filterFrontier,
                    graph->d_valuesPR, graph->d_degree, graph->d_sum);
              }
            }
          }
        }

        while (!targetGPUQueue.empty()) {

          uint32 tStream = targetGPUQueue.front();

          cudaStreamSynchronize(streams[tStream]);
          targetGPUQueue.pop();
          numPartitionsOnTarget++;

          PR32_Filter_Kernel<<<staticGrid, blockDim, 0, streams[tStream]>>>(
              &graph->d_partitionList[tStream], graph->d_partitionsOffsets,
              graph->d_values, graph->d_frontier, graph->d_filterEdges[tStream],
              graph->d_offsets, graph->d_filterFrontier, graph->d_valuesPR,
              graph->d_degree, graph->d_sum);
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

            cudaStreamSynchronize(neighborMemCpyStreams[gpu][nStream]);

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
              PR32_Static_NeighborFilter_Kernel<<<
                  staticGrid, blockDim, 0,
                  neighborComputeStreams[gpu][nStream]>>>(
                  &graph->d_nPartList[gpu][nStream], graph->d_partitionsOffsets,
                  graph->d_values, graph->d_frontier, graph->d_staticEdges,
                  graph->d_offsets, graph->d_filterFrontier, graph->d_valuesPR,
                  graph->d_degree, graph->d_sum);
            } else {
              PR32_NeighborFilter_Kernel<<<
                  staticGrid, blockDim, 0,
                  neighborComputeStreams[gpu][nStream]>>>(
                  &graph->d_nPartList[gpu][nStream], graph->d_partitionsOffsets,
                  graph->d_values, graph->d_frontier,
                  graph->d_nFilterEdges[gpu][nStream], graph->d_offsets,
                  graph->d_filterFrontier, graph->d_valuesPR, graph->d_degree,
                  graph->d_sum);
            }
          }
        }

        // std::cout << "Partitions processed in target GPU: "
        //           << numPartitionsOnTarget << std::endl;
        //
        // std::cout << "Partitions to be processed in neighbor GPUs: "
        //           << numPartitionsOnNeighbors << std::endl;
      }

      cudaDeviceSynchronize();

      PR32_Update_Values<<<staticGrid, blockDim>>>(
          graph->numVertices, graph->d_valuesPR, graph->d_sum,
          graph->d_frontier);

      cudaDeviceSynchronize();

      *(graph->frontierSize) = thrust::reduce(
          graph->thrustFrontier, graph->thrustFrontier + *(graph->numVertices),
          0, thrust::plus<uint32>());
    }
  }

  const uint64 partitionSizeMB = PARTITION_SIZE_MB / (1024 * 1024); // 1024^2

  uint64 MBytes = totalNumFilterPartitions * partitionSizeMB;

  // uint64 GBytes = MBytes >> 10;
  std::cout << "Total partitions in filter: " << totalNumFilterPartitions
            << std::endl;

  std::cout << "Total amount of data sent with filter: " << MBytes << " MB"
            << std::endl;

  graph->DumpValues();
  return;
}

void PR64(std::string filePath, uint32 nRuns) {
  ALGORITHM_TYPE algo = PR;
  CSR<uint64> *graph = new CSR<uint64>;
  graph->ReadInputFile(filePath, algo);

  // Adjust this number of blocks in x dimension to be a multiple of the number
  // of SMS and acquire better load balancing
  int device = 0; // Selected device
  uint64 k =
      4; // Multiple of SMs to choose for the grid dimension (to be adjusted)

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);
  uint64 numSMs = prop.multiProcessorCount;

  dim3 staticGrid = dim3(k * numSMs, 1, 1);

  dim3 blockDim(THREADS_PER_BLOCK, 1, 1); // (x,y,z) = (THREADS_PER_BLOCK, 1, 1)

  cudaStream_t staticStream, demandStream;
  GPUAssert(cudaStreamCreate(&staticStream));
  GPUAssert(cudaStreamCreate(&demandStream));

  // graph->InitData(0);

  GPUAssert(cudaPeekAtLastError());

  std::cout << "Starting Traversals" << std::endl;

  for (int test = 0; test < nRuns; test++) {

    graph->ResetFrontierNValues();

    *(graph->frontierSize) = thrust::reduce(
        graph->thrustFrontier, graph->thrustFrontier + *(graph->numVertices), 0,
        thrust::plus<uint64>());

    while (*(graph->frontierSize)) {

      // This kernel will set the active vertices label to either Static or
      // On-Demand
      setStaticNDemandFrontiers<<<staticGrid, blockDim, 0, staticStream>>>(
          graph->numVertices, graph->d_frontier, graph->d_staticFrontier,
          graph->d_demandFrontier, graph->d_inStatic);

      // Calculate the amount of active nodes in GPU memory
      *(graph->staticSize) =
          thrust::reduce(graph->thurstStaticFrontier,
                         graph->thurstStaticFrontier + *(graph->numVertices), 0,
                         thrust::plus<uint64>());

      if (*(graph->staticSize) > 0) {

        // thrust::device_ptr<uint64> thurstPrefixSum =
        // thrust::device_ptr<uint64>(graph->d_prefixSum);

        thrust::exclusive_scan(
            graph->thurstStaticFrontier,
            graph->thurstStaticFrontier + *(graph->numVertices),
            graph->thurstPrefixSum, 0, thrust::plus<uint64>());

        setStaticList<<<staticGrid, blockDim, 0, staticStream>>>(
            graph->numVertices, graph->d_staticList, graph->d_staticFrontier,
            graph->d_prefixSum);
      }

      // Calculate the amount of active vertices on-demand
      *(graph->demandSize) =
          thrust::reduce(graph->thurstDemandFrontier,
                         graph->thurstDemandFrontier + *(graph->numVertices), 0,
                         thrust::plus<uint64>());

      if (*(graph->demandSize) > 0) {
        // thrust::device_ptr<uint64> thurstPrefixSum =
        // thrust::device_ptr<uint64>(graph->d_prefixSum);

        thrust::exclusive_scan(
            graph->thurstDemandFrontier,
            graph->thurstDemandFrontier + *(graph->numVertices),
            graph->thurstPrefixSum, 0, thrust::plus<uint64>());

        setDemandList<<<staticGrid, blockDim, 0, demandStream>>>(
            graph->numVertices, graph->d_demandList, graph->d_demandFrontier,
            graph->d_prefixSum);
      }

      // setFrontierUnified<<<staticGrid, blockDim, 0,
      // staticStream>>>(graph->staticSize,
      //                                                               graph->d_staticList,
      //                                                               graph->demandSize,
      //                                                               graph->d_demandList,
      //                                                               graph->d_frontier);

      cudaStreamSynchronize(staticStream);

      // Test this in the upper branch ?? And we're using d_offsets instead of a
      // static h_offsets
      PR64_Static_Kernel<<<staticGrid, blockDim, 0, staticStream>>>(
          graph->staticSize, graph->d_staticList, graph->d_offsets,
          graph->d_staticEdges, graph->d_valuesPR, graph->d_degree,
          graph->d_sum, graph->d_inStatic);

      // GPUAssert(cudaPeekAtLastError());

      if (*(graph->demandSize) > 0) {
        uint64 numBlocks =
            (((*(graph->demandSize)) * WARP_SIZE + THREADS_PER_BLOCK) /
             THREADS_PER_BLOCK);
        dim3 gridDim(THREADS_PER_BLOCK,
                     (numBlocks + THREADS_PER_BLOCK) /
                         THREADS_PER_BLOCK); // (x,y,z) = (THREADS_PER_BLOCK, k
                                             // * THREADS_PER_BLOCK, 1)

        cudaStreamSynchronize(demandStream);

        PR64_Demand_Kernel<<<gridDim, blockDim, 0, demandStream>>>(
            graph->demandSize, graph->d_demandList, graph->h_edges,
            graph->d_offsets, graph->d_valuesPR, graph->d_degree, graph->d_sum,
            graph->d_inStatic);

        cudaStreamSynchronize(demandStream);
        cudaStreamSynchronize(staticStream);
      } else
        cudaDeviceSynchronize();

      // GPUAssert(cudaPeekAtLastError());

      PR64_Update_Values<<<staticGrid, blockDim>>>(
          graph->numVertices, graph->d_valuesPR, graph->d_sum,
          graph->d_frontier);

      cudaDeviceSynchronize();

      *(graph->frontierSize) = thrust::reduce(
          graph->thrustFrontier, graph->thrustFrontier + *(graph->numVertices),
          0, thrust::plus<uint64>());
    }
  }

  // We're gonna need to compare results now!!
  cudaMemcpy(graph->h_valuesPR, graph->d_valuesPR,
             *(graph->numVertices) * sizeof(double), cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();
  for (uint64 i = 0; i < 10; i++)
    std::cout << "Our result: " << graph->h_valuesPR[i] << std::endl;

  std::string filepath = "results/values1.bin";

  std::ofstream file(filepath);
  if (!file.is_open())
    return;

  // Write values
  for (uint64 i = 0; i < *(graph->numVertices); ++i)
    file.write(reinterpret_cast<const char *>(&graph->h_valuesPR[i]),
               sizeof(double));

  file.close();

  // graph->DumpValues();
  return;
}

void PR32_PUSH(std::string filePath, uint32 nRuns, uint32 nNeighborGPUs) {

  ALGORITHM_TYPE algo = PR;
  CSR<uint32> *graph = new CSR<uint32>;
  graph->ReadInputFile(filePath, algo);

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

  int device = 0;
  uint32 k = 2;

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);
  uint32 numSMs = prop.multiProcessorCount;

  dim3 staticGrid = dim3(k * numSMs, 1, 1);
  dim3 blockDim(THREADS_PER_BLOCK, 1, 1); // (x,y,z) = (THREADS_PER_BLOCK, 1, 1)

  GPUAssert(cudaPeekAtLastError());

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

      // std::cout << "Static size: " << *graph->staticSize << std::endl;
      if (*graph->frontierSize > 80 * graph->avgVertPerPart) {
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

        PR32_Static_Kernel_PUSH<<<staticGrid, blockDim, 0, staticStream>>>(
            graph->staticSize, graph->d_staticList, graph->d_offsets,
            graph->d_staticEdges, graph->d_frontier, graph->d_inStatic,
            graph->d_delta, graph->d_residual);
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

        PR32_Demand_Kernel_PUSH<<<gridDim, blockDim, 0, demandStream>>>(
            graph->demandSize, graph->d_demandList, graph->d_frontier,
            graph->h_edges, graph->d_offsets, graph->d_delta,
            graph->d_residual);
      }

      if (*graph->frontierSize > 80 * graph->avgVertPerPart) {
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
          cudaMemcpyAsync(graph->d_filterEdges[stream], &graph->h_edges[start],
                          partitionSize * sizeof(*graph->h_edges),
                          cudaMemcpyDefault, streams[stream]);

          cudaMemcpyAsync(&graph->d_partitionList[stream],
                          &graph->h_partitionList[stream],
                          sizeof(*graph->h_partitionList),
                          cudaMemcpyHostToDevice, streams[stream]);

          // GPUAssert(cudaPeekAtLastError());
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

            cudaMemcpyAsync(graph->d_nFilterEdges[gpu][neighborStream],
                            (gpu > 0) ? graph->h_edges + neighborStart
                                      : graph->h_edges + neighborStart,
                            neighborPartitionSize * sizeof(*graph->h_edges),
                            cudaMemcpyDefault,
                            neighborMemCpyStreams[gpu][neighborStream]);

            // GPUAssert(cudaPeekAtLastError());
            // We can prob allocate this data in the other numa node too
            cudaMemcpyAsync(&graph->d_nPartList[gpu][neighborStream],
                            graph->h_nPartList[gpu] + neighborStream,
                            sizeof(**graph->h_nPartList),
                            cudaMemcpyHostToDevice,
                            neighborMemCpyStreams[gpu][neighborStream]);

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
            PR32_Filter_Kernel_PUSH<<<staticGrid, blockDim, 0,
                                      streams[tStream]>>>(
                &graph->d_partitionList[tStream], graph->d_partitionsOffsets,
                graph->d_values, graph->d_frontier,
                graph->d_filterEdges[tStream], graph->d_offsets,
                graph->d_filterFrontier, graph->d_valuesPR, graph->d_residual,
                graph->d_delta);

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
              // PR32_NeighborFilter_Kernel_PUSH<<<
              //     staticGrid, blockDim, 0,
              //     neighborComputeStreams[gpu][nStream]>>>(
              //     &graph->d_nPartList[gpu][nStream],
              //     graph->d_partitionsOffsets, graph->d_values,
              //     graph->d_frontier, graph->d_nFilterEdges[gpu][nStream],
              //     graph->d_offsets, graph->d_filterFrontier,
              //     graph->d_valuesPR, graph->d_residual, graph->d_delta);
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
                PR32_Static_Filter_Kernel_PUSH<<<
                    staticGrid, blockDim, 0,
                    neighborComputeStreams[gpu][nStream]>>>(
                    &graph->d_nPartList[gpu][nStream],
                    graph->d_partitionsOffsets, graph->d_values,
                    graph->d_frontier, graph->d_staticEdges, graph->d_offsets,
                    graph->d_filterFrontier, graph->d_valuesPR,
                    graph->d_residual, graph->d_delta);
              } else {
                PR32_NeighborFilter_Kernel_PUSH<<<
                    staticGrid, blockDim, 0,
                    neighborComputeStreams[gpu][nStream]>>>(
                    &graph->d_nPartList[gpu][nStream],
                    graph->d_partitionsOffsets, graph->d_values,
                    graph->d_frontier, graph->d_nFilterEdges[gpu][nStream],
                    graph->d_offsets, graph->d_filterFrontier,
                    graph->d_valuesPR, graph->d_residual, graph->d_delta);
              }
            }
          }
        }

        while (!targetGPUQueue.empty()) {

          uint32 tStream = targetGPUQueue.front();

          cudaStreamSynchronize(streams[tStream]);
          targetGPUQueue.pop();
          numPartitionsOnTarget++;

          PR32_Filter_Kernel_PUSH<<<staticGrid, blockDim, 0,
                                    streams[tStream]>>>(
              &graph->d_partitionList[tStream], graph->d_partitionsOffsets,
              graph->d_values, graph->d_frontier, graph->d_filterEdges[tStream],
              graph->d_offsets, graph->d_filterFrontier, graph->d_valuesPR,
              graph->d_residual, graph->d_delta);
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

          // std::cout << "Nao enviei uma partition certinho" << std::endl;
        }

        for (uint32 gpu = 0; gpu < neighborGPUQueues.size(); gpu++) {

          while (!neighborGPUQueues[gpu].empty()) {

            // std::cout << "Nao enviei uma partition certinho" << std::endl;
            uint32 nStream = neighborGPUQueues[gpu].front();

            cudaSetDevice(gpu + 1);

            cudaStreamSynchronize(neighborMemCpyStreams[gpu][nStream]);

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
              cudaMemcpyAsync(&graph->d_staticEdges[partitionStart],
                              graph->d_nFilterEdges[gpu][nStream],
                              processedPartitionSize * sizeof(*graph->h_edges),
                              cudaMemcpyDeviceToDevice,
                              neighborComputeStreams[gpu][nStream]);

              cudaMemsetAsync(
                  &graph->d_inStatic
                       [graph->h_partitionsOffsets[processedPartition]],
                  1,
                  (graph->h_partitionsOffsets[processedPartition + 1] -
                   graph->h_partitionsOffsets[processedPartition]) *
                      sizeof(*graph->d_inStatic),
                  neighborComputeStreams[gpu][nStream]);

              // cudaDeviceSynchronize();
              PR32_Static_Filter_Kernel_PUSH<<<
                  staticGrid, blockDim, 0,
                  neighborComputeStreams[gpu][nStream]>>>(
                  &graph->d_nPartList[gpu][nStream], graph->d_partitionsOffsets,
                  graph->d_values, graph->d_frontier, graph->d_staticEdges,
                  graph->d_offsets, graph->d_filterFrontier, graph->d_valuesPR,
                  graph->d_residual, graph->d_delta);
            } else {
              PR32_NeighborFilter_Kernel_PUSH<<<
                  staticGrid, blockDim, 0,
                  neighborComputeStreams[gpu][nStream]>>>(
                  &graph->d_nPartList[gpu][nStream], graph->d_partitionsOffsets,
                  graph->d_values, graph->d_frontier,
                  graph->d_nFilterEdges[gpu][nStream], graph->d_offsets,
                  graph->d_filterFrontier, graph->d_valuesPR, graph->d_residual,
                  graph->d_delta);
            }
          }
        }

        // std::cout << "Partitions processed in target GPU: "
        //           << numPartitionsOnTarget << std::endl;
        //
        // std::cout << "Partitions to be processed in neighbor GPUs: "
        //           << numPartitionsOnNeighbors << std::endl;
      }
      cudaDeviceSynchronize();

      // GPUAssert(cudaPeekAtLastError());

      PR32_Update_Values_PUSH<<<staticGrid, blockDim>>>(
          graph->numVertices, graph->d_valuesPR, graph->d_frontier,
          graph->d_offsets, graph->d_delta, graph->d_residual);

      cudaDeviceSynchronize();

      *(graph->frontierSize) = thrust::reduce(
          graph->thrustFrontier, graph->thrustFrontier + *(graph->numVertices),
          0, thrust::plus<uint32>());
    }
  }

  const uint64 partitionSizeMB = PARTITION_SIZE_MB / (1024 * 1024); // 1024^2

  uint64 MBytes = totalNumFilterPartitions * partitionSizeMB;

  // uint64 GBytes = MBytes >> 10;
  std::cout << "Total partitions in filter: " << totalNumFilterPartitions
            << std::endl;

  std::cout << "Total amount of data sent with filter: " << MBytes << " MB"
            << std::endl;

  graph->DumpValues();
  return;
}
