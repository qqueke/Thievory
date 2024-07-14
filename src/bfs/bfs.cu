#include "bfs.cuh"
#include <iostream>
#include <numa.h>
#include <ostream>
#include <queue>
#include <vector>

#define N_FILTER_STREAMS2 128

// Test the order between static and demand kernel
void BFS32(string filePath, uint32 srcVertex, double memAdvise, uint32 nRuns,
           uint32 nNeighborGPUs) {

  numa_run_on_node(0);
  ALGORITHM_TYPE algo = BFS;
  CSR<uint32> *graph = new CSR<uint32>;
  graph->ReadInputFile(filePath, algo);
  graph->InitData(srcVertex, nNeighborGPUs);
  // Adjust this number of blocks in x dimension to be a multiple of the number
  // of SMS and acquire better load balancing
  int device = 0; // Selected device
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

      if (*graph->frontierSize > 1000000) {
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

        BFS32_Static_Kernel<<<staticGrid, blockDim, 0, staticStream>>>(
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

        BFS32_Demand_Kernel<<<gridDim, blockDim, 0, demandStream>>>(
            graph->demandSize, graph->d_demandList, graph->d_values,
            graph->d_frontier, graph->h_edges, graph->d_offsets);
      }

      if (*graph->frontierSize > 1000000) {

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
            BFS32_Filter_Kernel<<<staticGrid, blockDim, 0, streams[tStream]>>>(
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

              BFS32_NeighborFilter_Kernel<<<
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
          BFS32_Filter_Kernel<<<staticGrid, blockDim, 0, streams[tStream]>>>(
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

            BFS32_NeighborFilter_Kernel<<<
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

  cudaDeviceSynchronize();

  for (uint32 i = 0; i < 31; i++) {
    std::cout << "Our result: " << graph->h_values[i] << std::endl;
  }

  graph->DumpValues();
  return;
}

void BFS64(string filePath, uint32 srcVertex, double memAdvise, uint32 nRuns) {

  return;
}
