#include "pr.cuh"

void PR32(string filePath, double memAdvise, uint32 nRuns)
{
    ALGORITHM_TYPE algo = PR;
    CSR<uint32> *graph = new CSR<uint32>;
    graph->ReadInputFile(filePath, algo);

    // Adjust this number of blocks in x dimension to be a multiple of the number of SMS and acquire better load balancing
    int device = 0; // Selected device
    uint32 k = 4;   // Multiple of SMs to choose for the grid dimension (to be adjusted)

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    uint32 numSMs = prop.multiProcessorCount;

    dim3 staticGrid = dim3(k * numSMs, 1, 1);

    dim3 blockDim(THREADS_PER_BLOCK, 1, 1); // (x,y,z) = (THREADS_PER_BLOCK, 1, 1)

    cudaStream_t staticStream, demandStream;
    GPUAssert(cudaStreamCreate(&staticStream));
    GPUAssert(cudaStreamCreate(&demandStream));

    graph->InitData(0);

    GPUAssert(cudaPeekAtLastError());

    std::cout << "Starting Traversals" << std::endl;
    TimeRecord<chrono::milliseconds> totalProcess("Total execution");

    for (int test = 0; test < nRuns; test++)
    {

        graph->ResetFrontierNValues();

        *(graph->frontierSize) = thrust::reduce(graph->thrustFrontier,
                                                graph->thrustFrontier + *(graph->numVertices),
                                                0,
                                                thrust::plus<uint32>());

        totalProcess.startRecord();
        while (*(graph->frontierSize))
        {

            // This kernel will set the active vertices label to either Static or On-Demand
            setStaticNDemandFrontiers<<<staticGrid, blockDim, 0, staticStream>>>(graph->numVertices,
                                                                                 graph->d_frontier,
                                                                                 graph->d_staticFrontier,
                                                                                 graph->d_demandFrontier,
                                                                                 graph->d_inStatic);

            // Calculate the amount of active nodes in GPU memory
            *(graph->staticSize) = thrust::reduce(graph->thurstStaticFrontier,
                                                  graph->thurstStaticFrontier + *(graph->numVertices),
                                                  0,
                                                  thrust::plus<uint32>());

            if (*(graph->staticSize) > 0)
            {

                // thrust::device_ptr<uint64> thurstPrefixSum = thrust::device_ptr<uint64>(graph->d_prefixSum);

                thrust::exclusive_scan(graph->thurstStaticFrontier,
                                       graph->thurstStaticFrontier + *(graph->numVertices),
                                       graph->thurstPrefixSum,
                                       0,
                                       thrust::plus<uint32>());

                setStaticList<<<staticGrid, blockDim, 0, staticStream>>>(graph->numVertices,
                                                                         graph->d_staticList,
                                                                         graph->d_staticFrontier,
                                                                         graph->d_prefixSum);
            }

            // Calculate the amount of active vertices on-demand
            *(graph->demandSize) = thrust::reduce(graph->thurstDemandFrontier,
                                                  graph->thurstDemandFrontier + *(graph->numVertices),
                                                  0,
                                                  thrust::plus<uint32>());

            if (*(graph->demandSize) > 0)
            {
                // thrust::device_ptr<uint64> thurstPrefixSum = thrust::device_ptr<uint64>(graph->d_prefixSum);

                thrust::exclusive_scan(graph->thurstDemandFrontier,
                                       graph->thurstDemandFrontier + *(graph->numVertices),
                                       graph->thurstPrefixSum,
                                       0,
                                       thrust::plus<uint32>());

                setDemandList<<<staticGrid, blockDim, 0, demandStream>>>(graph->numVertices,
                                                                         graph->d_demandList,
                                                                         graph->d_demandFrontier,
                                                                         graph->d_prefixSum);
            }

            // setFrontierUnified<<<staticGrid, blockDim, 0, staticStream>>>(graph->staticSize,
            //                                                               graph->d_staticList,
            //                                                               graph->demandSize,
            //                                                               graph->d_demandList,
            //                                                               graph->d_frontier);

            cudaStreamSynchronize(staticStream);

            // Test this in the upper branch ?? And we're using d_offsets instead of a static h_offsets
            PR32_Static_Kernel<<<staticGrid, blockDim, 0, staticStream>>>(graph->staticSize,
                                                                          graph->d_staticList,
                                                                          graph->d_offsets,
                                                                          graph->d_staticEdges,
                                                                          graph->d_valuesPR,
                                                                          graph->d_degree,
                                                                          graph->d_sum,
                                                                          graph->d_inStatic);

            // GPUAssert(cudaPeekAtLastError());

            if (*(graph->demandSize) > 0)
            {
                uint32 numBlocks = (((*(graph->demandSize)) * WARP_SIZE + THREADS_PER_BLOCK) / THREADS_PER_BLOCK);
                dim3 gridDim(THREADS_PER_BLOCK, (numBlocks + THREADS_PER_BLOCK) / THREADS_PER_BLOCK); // (x,y,z) = (THREADS_PER_BLOCK, k * THREADS_PER_BLOCK, 1)

                cudaStreamSynchronize(demandStream);

                PR32_Demand_Kernel<<<gridDim, blockDim, 0, demandStream>>>(graph->demandSize,
                                                                           graph->d_demandList,
                                                                           graph->h_edges,
                                                                           graph->d_offsets,
                                                                           graph->d_valuesPR,
                                                                           graph->d_degree,
                                                                           graph->d_sum,
                                                                           graph->d_inStatic);

                cudaStreamSynchronize(demandStream);
                cudaStreamSynchronize(staticStream);
            }
            else
                cudaDeviceSynchronize();

            // GPUAssert(cudaPeekAtLastError());

            PR32_Update_Values<<<staticGrid, blockDim>>>(graph->numVertices,
                                                         graph->d_valuesPR,
                                                         graph->d_sum,
                                                         graph->d_frontier);

            cudaDeviceSynchronize();

            *(graph->frontierSize) = thrust::reduce(graph->thrustFrontier,
                                                    graph->thrustFrontier + *(graph->numVertices),
                                                    0,
                                                    thrust::plus<uint32>());
        }
    }

    totalProcess.endRecord();
    totalProcess.print();

    // We're gonna need to compare results now!!
    cudaMemcpy(graph->h_valuesPR, graph->d_valuesPR, *(graph->numVertices) * sizeof(double), cudaMemcpyDeviceToHost);

    for (uint32 i = 0; i < 10; i++)
        std::cout << "Our result: " << graph->h_valuesPR[i] << std::endl;

    // We're gonna need to compare results now!!
    cudaMemcpy(graph->h_valuesPR, graph->d_valuesPR, *(graph->numVertices) * sizeof(double), cudaMemcpyDeviceToHost);

    for (uint64 i = 0; i < 10; i++)
        std::cout << "Our result: " << graph->h_valuesPR[i] << std::endl;

    std::string filepath = "results/values1.bin";

    std::ofstream file(filepath);
    if (!file.is_open())
        return;

    // Write values
    for (uint64 i = 0; i < *(graph->numVertices); ++i)
        file.write(reinterpret_cast<const char *>(&graph->h_valuesPR[i]), sizeof(double));

    file.close();

    // graph->DumpValues();
    return;
}

void PR64(string filePath, double memAdvise, uint32 nRuns)
{
    ALGORITHM_TYPE algo = PR;
    CSR<uint64> *graph = new CSR<uint64>;
    graph->ReadInputFile(filePath, algo);

    // Adjust this number of blocks in x dimension to be a multiple of the number of SMS and acquire better load balancing
    int device = 0; // Selected device
    uint64 k = 4;   // Multiple of SMs to choose for the grid dimension (to be adjusted)

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    uint64 numSMs = prop.multiProcessorCount;

    dim3 staticGrid = dim3(k * numSMs, 1, 1);

    dim3 blockDim(THREADS_PER_BLOCK, 1, 1); // (x,y,z) = (THREADS_PER_BLOCK, 1, 1)

    cudaStream_t staticStream, demandStream;
    GPUAssert(cudaStreamCreate(&staticStream));
    GPUAssert(cudaStreamCreate(&demandStream));

    graph->InitData(0);

    GPUAssert(cudaPeekAtLastError());

    std::cout << "Starting Traversals" << std::endl;
    TimeRecord<chrono::milliseconds> totalProcess("Total execution");

    for (int test = 0; test < nRuns; test++)
    {

        graph->ResetFrontierNValues();

        *(graph->frontierSize) = thrust::reduce(graph->thrustFrontier,
                                                graph->thrustFrontier + *(graph->numVertices),
                                                0,
                                                thrust::plus<uint64>());

        totalProcess.startRecord();
        while (*(graph->frontierSize))
        {

            // This kernel will set the active vertices label to either Static or On-Demand
            setStaticNDemandFrontiers<<<staticGrid, blockDim, 0, staticStream>>>(graph->numVertices,
                                                                                 graph->d_frontier,
                                                                                 graph->d_staticFrontier,
                                                                                 graph->d_demandFrontier,
                                                                                 graph->d_inStatic);

            // Calculate the amount of active nodes in GPU memory
            *(graph->staticSize) = thrust::reduce(graph->thurstStaticFrontier,
                                                  graph->thurstStaticFrontier + *(graph->numVertices),
                                                  0,
                                                  thrust::plus<uint64>());

            if (*(graph->staticSize) > 0)
            {

                // thrust::device_ptr<uint64> thurstPrefixSum = thrust::device_ptr<uint64>(graph->d_prefixSum);

                thrust::exclusive_scan(graph->thurstStaticFrontier,
                                       graph->thurstStaticFrontier + *(graph->numVertices),
                                       graph->thurstPrefixSum,
                                       0,
                                       thrust::plus<uint64>());

                setStaticList<<<staticGrid, blockDim, 0, staticStream>>>(graph->numVertices,
                                                                         graph->d_staticList,
                                                                         graph->d_staticFrontier,
                                                                         graph->d_prefixSum);
            }

            // Calculate the amount of active vertices on-demand
            *(graph->demandSize) = thrust::reduce(graph->thurstDemandFrontier,
                                                  graph->thurstDemandFrontier + *(graph->numVertices),
                                                  0,
                                                  thrust::plus<uint64>());

            if (*(graph->demandSize) > 0)
            {
                // thrust::device_ptr<uint64> thurstPrefixSum = thrust::device_ptr<uint64>(graph->d_prefixSum);

                thrust::exclusive_scan(graph->thurstDemandFrontier,
                                       graph->thurstDemandFrontier + *(graph->numVertices),
                                       graph->thurstPrefixSum,
                                       0,
                                       thrust::plus<uint64>());

                setDemandList<<<staticGrid, blockDim, 0, demandStream>>>(graph->numVertices,
                                                                         graph->d_demandList,
                                                                         graph->d_demandFrontier,
                                                                         graph->d_prefixSum);
            }

            // setFrontierUnified<<<staticGrid, blockDim, 0, staticStream>>>(graph->staticSize,
            //                                                               graph->d_staticList,
            //                                                               graph->demandSize,
            //                                                               graph->d_demandList,
            //                                                               graph->d_frontier);

            cudaStreamSynchronize(staticStream);

            // Test this in the upper branch ?? And we're using d_offsets instead of a static h_offsets
            PR64_Static_Kernel<<<staticGrid, blockDim, 0, staticStream>>>(graph->staticSize,
                                                                          graph->d_staticList,
                                                                          graph->d_offsets,
                                                                          graph->d_staticEdges,
                                                                          graph->d_valuesPR,
                                                                          graph->d_degree,
                                                                          graph->d_sum,
                                                                          graph->d_inStatic);

            __global__ void PR64_Static_Kernel(const uint64 *staticSize, const uint64 *d_staticList, const uint64 *d_offsets, const uint64 *d_staticEdges, bool *d_frontier, const bool *d_inStatic, double *d_delta, double *d_residual);

            // GPUAssert(cudaPeekAtLastError());

            if (*(graph->demandSize) > 0)
            {
                uint64 numBlocks = (((*(graph->demandSize)) * WARP_SIZE + THREADS_PER_BLOCK) / THREADS_PER_BLOCK);
                dim3 gridDim(THREADS_PER_BLOCK, (numBlocks + THREADS_PER_BLOCK) / THREADS_PER_BLOCK); // (x,y,z) = (THREADS_PER_BLOCK, k * THREADS_PER_BLOCK, 1)

                cudaStreamSynchronize(demandStream);

                PR64_Demand_Kernel<<<gridDim, blockDim, 0, demandStream>>>(graph->demandSize,
                                                                           graph->d_demandList,
                                                                           graph->h_edges,
                                                                           graph->d_offsets,
                                                                           graph->d_valuesPR,
                                                                           graph->d_degree,
                                                                           graph->d_sum,
                                                                           graph->d_inStatic);

                cudaStreamSynchronize(demandStream);
                cudaStreamSynchronize(staticStream);
            }
            else
                cudaDeviceSynchronize();

            // GPUAssert(cudaPeekAtLastError());

            PR64_Update_Values<<<staticGrid, blockDim>>>(graph->numVertices,
                                                         graph->d_valuesPR,
                                                         graph->d_sum,
                                                         graph->d_frontier);

            cudaDeviceSynchronize();

            *(graph->frontierSize) = thrust::reduce(graph->thrustFrontier,
                                                    graph->thrustFrontier + *(graph->numVertices),
                                                    0,
                                                    thrust::plus<uint64>());
        }
    }

    totalProcess.endRecord();
    totalProcess.print();

    // We're gonna need to compare results now!!
    cudaMemcpy(graph->h_valuesPR, graph->d_valuesPR, *(graph->numVertices) * sizeof(double), cudaMemcpyDeviceToHost);

    for (uint64 i = 0; i < 10; i++)
        std::cout << "Our result: " << graph->h_valuesPR[i] << std::endl;

    std::string filepath = "results/values1.bin";

    std::ofstream file(filepath);
    if (!file.is_open())
        return;

    // Write values
    for (uint64 i = 0; i < *(graph->numVertices); ++i)
        file.write(reinterpret_cast<const char *>(&graph->h_valuesPR[i]), sizeof(double));

    file.close();

    // graph->DumpValues();
    return;
}

void PR32_PUSH(string filePath, double memAdvise, uint32 nRuns)
{
    ALGORITHM_TYPE algo = PR;
    CSR<uint32> *graph = new CSR<uint32>;
    graph->ReadInputFile(filePath, algo);

    // Adjust this number of blocks in x dimension to be a multiple of the number of SMS and acquire better load balancing
    int device = 0; // Selected device
    uint32 k = 2;   // Multiple of SMs to choose for the grid dimension (to be adjusted)

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    uint32 numSMs = prop.multiProcessorCount;

    dim3 staticGrid = dim3(k * numSMs, 1, 1);
    // dim3 staticGrid = dim3(56, 1, 1);

    dim3 blockDim(THREADS_PER_BLOCK, 1, 1); // (x,y,z) = (THREADS_PER_BLOCK, 1, 1)

    cudaStream_t staticStream, demandStream;
    GPUAssert(cudaStreamCreate(&staticStream));
    GPUAssert(cudaStreamCreate(&demandStream));

    double *h_delta = new double[*graph->numVertices];
    double *h_residual = new double[*graph->numVertices];

    GPUAssert(cudaMalloc(&graph->d_delta, (*graph->numVertices + 1) * sizeof(*graph->d_delta)));
    GPUAssert(cudaMalloc(&graph->d_residual, (*graph->numVertices + 1) * sizeof(*graph->d_residual)));

    graph->InitData(0);

    GPUAssert(cudaPeekAtLastError());

    std::cout << "Starting Traversals" << std::endl;
    TimeRecord<chrono::milliseconds> totalProcess("Total execution");

    for (int test = 0; test < nRuns; test++)
    {

        graph->ResetFrontierNValues();

        for (uint32 i = 0; i < *graph->numVertices; i++)
        {
            graph->h_valuesPR[i] = (double)(1.0f - ALPHA);
            h_delta[i] = (double)((1.0f - ALPHA) * ALPHA / (graph->h_offsets[i + 1] - graph->h_offsets[i]));
            h_residual[i] = (double)0.0f;
        }
        cudaMemcpy(graph->d_valuesPR, graph->h_valuesPR, *graph->numVertices * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(graph->d_delta, h_delta, *graph->numVertices * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(graph->d_residual, h_residual, *graph->numVertices * sizeof(double), cudaMemcpyHostToDevice);

        *(graph->frontierSize) = thrust::reduce(graph->thrustFrontier,
                                                graph->thrustFrontier + *(graph->numVertices),
                                                0,
                                                thrust::plus<uint32>());

        totalProcess.startRecord();
        while (*(graph->frontierSize))
        {

            // This kernel will set the active vertices label to either Static or On-Demand
            setStaticNDemandFrontiers<<<staticGrid, blockDim>>>(graph->numVertices,
                                                                graph->d_frontier,
                                                                graph->d_staticFrontier,
                                                                graph->d_demandFrontier,
                                                                graph->d_inStatic);

            // Calculate the amount of active nodes in GPU memory
            *(graph->staticSize) = thrust::reduce(graph->thurstStaticFrontier,
                                                  graph->thurstStaticFrontier + *(graph->numVertices),
                                                  0,
                                                  thrust::plus<uint32>());

            if (*(graph->staticSize) > 0)
            {

                // thrust::device_ptr<uint64> thurstPrefixSum = thrust::device_ptr<uint64>(graph->d_prefixSum);

                thrust::exclusive_scan(graph->thurstStaticFrontier,
                                       graph->thurstStaticFrontier + *(graph->numVertices),
                                       graph->thurstPrefixSum,
                                       0,
                                       thrust::plus<uint32>());

                setStaticList<<<staticGrid, blockDim>>>(graph->numVertices,
                                                        graph->d_staticList,
                                                        graph->d_staticFrontier,
                                                        graph->d_prefixSum);
            }

            // Calculate the amount of active vertices on-demand
            *(graph->demandSize) = thrust::reduce(graph->thurstDemandFrontier,
                                                  graph->thurstDemandFrontier + *(graph->numVertices),
                                                  0,
                                                  thrust::plus<uint32>());

            if (*(graph->demandSize) > 0)
            {
                // thrust::device_ptr<uint64> thurstPrefixSum = thrust::device_ptr<uint64>(graph->d_prefixSum);

                thrust::exclusive_scan(graph->thurstDemandFrontier,
                                       graph->thurstDemandFrontier + *(graph->numVertices),
                                       graph->thurstPrefixSum,
                                       0,
                                       thrust::plus<uint32>());

                setDemandList<<<staticGrid, blockDim>>>(graph->numVertices,
                                                        graph->d_demandList,
                                                        graph->d_demandFrontier,
                                                        graph->d_prefixSum);
            }

            // Removes every static vertex from the current frontier
            if (*(graph->staticSize) > 0)
                setFrontier<<<staticGrid, blockDim, 0, staticStream>>>(graph->staticSize,
                                                                       graph->d_staticList,
                                                                       graph->d_frontier);

            // Removes every demand vertex from the current frontier
            if (*(graph->demandSize) > 0)
                setFrontier<<<staticGrid, blockDim, 0, staticStream>>>(graph->demandSize,
                                                                       graph->d_demandList,
                                                                       graph->d_frontier);

            // Test this in the upper branch ?? And we're using d_offsets instead of a static h_offsets
            PR32_Static_Kernel_PUSH<<<staticGrid, blockDim, 0, staticStream>>>(graph->staticSize,
                                                                               graph->d_staticList,
                                                                               graph->d_offsets,
                                                                               graph->d_staticEdges,
                                                                               graph->d_frontier,
                                                                               graph->d_inStatic,
                                                                               graph->d_delta,
                                                                               graph->d_residual);

            cudaDeviceSynchronize();

            if (*(graph->demandSize) > 0)
            {
                uint32 numBlocks = (((*(graph->demandSize)) * WARP_SIZE + THREADS_PER_BLOCK) / THREADS_PER_BLOCK);
                dim3 gridDim(THREADS_PER_BLOCK, (numBlocks + THREADS_PER_BLOCK) / THREADS_PER_BLOCK); // (x,y,z) = (THREADS_PER_BLOCK, k * THREADS_PER_BLOCK, 1)

                cudaStreamSynchronize(demandStream);

                PR32_Demand_Kernel_PUSH<<<gridDim, blockDim, 0, demandStream>>>(graph->demandSize,
                                                                                graph->d_demandList,
                                                                                graph->d_frontier,
                                                                                graph->h_edges,
                                                                                graph->d_offsets,
                                                                                graph->d_delta,
                                                                                graph->d_residual);

                cudaStreamSynchronize(demandStream);
                cudaStreamSynchronize(staticStream);
            }
            else
                cudaDeviceSynchronize();

            // GPUAssert(cudaPeekAtLastError());

            PR32_Update_Values_PUSH<<<staticGrid, blockDim, 0, staticStream>>>(graph->numVertices,
                                                                               graph->d_valuesPR,
                                                                               graph->d_frontier,
                                                                               graph->d_offsets,
                                                                               graph->d_delta,
                                                                               graph->d_residual);

            cudaStreamSynchronize(staticStream);

            *(graph->frontierSize) = thrust::reduce(graph->thrustFrontier,
                                                    graph->thrustFrontier + *(graph->numVertices),
                                                    0,
                                                    thrust::plus<uint32>());
        }
    }

    totalProcess.endRecord();
    totalProcess.print();

    // We're gonna need to compare results now!!
    cudaMemcpy(graph->h_valuesPR, graph->d_valuesPR, *(graph->numVertices) * sizeof(double), cudaMemcpyDeviceToHost);

    for (uint32 i = 0; i < 10; i++)
        std::cout << "Our result: " << graph->h_valuesPR[i] << std::endl;

    // graph->DumpValues();
    return;
}
