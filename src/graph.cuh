#ifndef GRAPH_CUH
#define GRAPH_CUH

#include "common.cuh"
#include "numa.h"
#include "timer.cuh"
template <class EdgeType> class CSR {
public:
  ALGORITHM_TYPE algorithm; // Chosen algorithm

  EdgeType *numVertices; // Total number of vertices transferred via Zero-Copy
  uint64 numEdges;       // Total number of edges
  uint64 numStaticEdges; // Number of edges in device memory

  EdgeType *frontierSize; // Redo this as a EdgeType
  EdgeType *staticSize;   // Static Size transferred via Zero-Copy
  EdgeType *demandSize;   // Demand Size transferred via Zero-Copy

  uint32 *h_partitionsOffsets; // Offsets array for partitions
  uint32 *d_partitionsOffsets; // Offsets array for partitions

  uint64 *h_offsets; // Vertices array
  uint64 *d_offsets; // Device vertices array

  EdgeType *h_edges;       // Edges in host memory
  EdgeType *d_staticEdges; // Edges in device memory

  EdgeType *h_edges2;        // Edges in host memory
  EdgeType *h_weights;       // Host weights
  EdgeType *d_staticWeights; // Device weights

  EdgeType *h_values; // Host values array
  EdgeType *d_values; // Device values array

  // Partition stuff
  uint32 *numPartitions;
  uint64 maxEdgesInPartition = std::floor(PARTITION_SIZE_MB / sizeof(EdgeType));

  // Filtering partitions data
  bool *h_filterFrontier;
  bool *d_filterFrontier; // Device Filter Frontier

  float *h_filterPartitionCost;
  float *d_filterPartitionCost; // Device transfer cost for partitions given
                                // filtering approach

  float *d_partitionCost; // Device transfer cost for partitions
                          // filtering approach

  float *h_partitionCost;

  uint32 *h_partition;
  uint32 *d_partitionNum;

  EdgeType
      *d_filterEdges[N_FILTER_STREAMS]; // Filter partition to be transferred
  EdgeType *d_filterWeights;            // Device weights

  EdgeType *d_neighborFilterEdges[N_FILTER_STREAMS];

  EdgeType *d_neighborFilterEdges2[N_FILTER_STREAMS];

  EdgeType *d_neighborFilterEdges3[N_FILTER_STREAMS];
  uint32 *h_filterOffset;
  uint32 *d_filterOffset;

  uint32 *h_partitionList;
  uint32 *h_neighborPartitionList;

  uint32 *h_neighborPartitionList3;
  uint32 *h_neighborPartitionList2;
  uint32 *d_partitionList;
  uint32 *d_neighborPartitionList;

  uint32 *d_neighborPartitionList3;
  uint32 *d_neighborPartitionList2;
  bool *h_finished;
  bool *d_finished;

  uint32 *h_nCoalescedPartitions;

  uint32 *d_nCoalescedPartitions;

  bool *h_inStatic; // Static bitmap
  bool *d_inStatic; // Data in shared memory

  bool *h_frontier;                        // Host Frontier
  bool *d_frontier;                        // Device Frontier
  thrust::device_ptr<bool> thrustFrontier; // Wrapper for Frontier

  bool *d_staticFrontier; // Device frontier in static memory
  EdgeType *d_staticList; // Device list for frontier vertices in static region
  thrust::device_ptr<bool> thurstStaticFrontier; // Wrapper for Static Frontier

  bool *d_demandFrontier; // Frontier inside host memory
  EdgeType *d_demandList; //  Device list for frontier vertices in host memory
  thrust::device_ptr<bool>
      thurstDemandFrontier; // Wrapper for On-Demand Frontier

  EdgeType *d_prefixSum; // Auxiliary array for Prefix Sum
  thrust::device_ptr<EdgeType> thurstPrefixSum; // Wrapper for Prefix Sum

  EdgeType srcVertex; // Source vertex for BFS

  uint32 *h_degree; // Pull PR specific
  uint32 *d_degree; // Pull PR specific

  double *h_valuesPR; // Host values array for each edge (PR only)
  double *d_valuesPR; // Device values array for each edge (PR only)

  double *d_sum;                          // PR only PUSH
  thrust::device_ptr<double> d_thurstSum; // PR only PUSH

  double *d_delta;    // PR only PUSH
  double *d_residual; // PR only PUSH

  // Reset frontier and values arrays
  void ResetFrontierNValues();

  // Initializes frontier and values arrays
  void InitFrontierNValues();

  // Sets the maximum partition size for edges in static memory
  void SetPartitionsConfig();

  // Sets the maximum partition size for edges in static memory
  void SetNumStaticEdges();

  // Initializes graph
  void InitData(uint64 sourceVertex);

  // Reads input file
  void ReadInputFile(const std::string &fileName, ALGORITHM_TYPE algo);

  // Dumps results to file
  void DumpValues();

  // Frees all allocated arrays in host and device
  void Free();
};

template <class EdgeType> void CSR<EdgeType>::ResetFrontierNValues() {
  if (algorithm == PR)
    cudaMemcpy(d_valuesPR, h_valuesPR, *numVertices * sizeof(*d_valuesPR),
               cudaMemcpyHostToDevice);
  else
    cudaMemcpy(d_values, h_values, *numVertices * sizeof(*d_values),
               cudaMemcpyHostToDevice);

  cudaMemcpy(d_frontier, h_frontier, *numVertices * sizeof(*d_frontier),
             cudaMemcpyHostToDevice);
  cudaMemset(d_staticFrontier, 0, *numVertices * sizeof(*d_staticFrontier));
  cudaMemset(d_demandFrontier, 0, *numVertices * sizeof(*d_demandFrontier));

  cudaMemset(d_filterFrontier, 0, *numVertices * sizeof(*d_filterFrontier));
  cudaDeviceSynchronize();

  return;
}

template <class EdgeType> void CSR<EdgeType>::InitFrontierNValues() {
  switch (algorithm) {
  case PR:

    for (uint32 i = 0; i < *numVertices; i++) {
      h_frontier[i] = 1;
      h_valuesPR[i] = 1.0 / (*numVertices);
    }
    break;
  case BFS:
    for (uint32 i = 0; i < *numVertices; i++) {
      h_frontier[i] = 0;
      h_values[i] = *numVertices + 1;
    }
    h_frontier[srcVertex] = 1;
    h_values[srcVertex] = 1;
    break;
  case SSSP:
    for (uint32 i = 0; i < *numVertices; i++) {
      h_frontier[i] = 0;
      h_values[i] = *numVertices + 1;
    }
    h_frontier[srcVertex] = 1;
    h_values[srcVertex] = 1;
    break;
  case CC:
    for (uint32 i = 0; i < *numVertices; i++) {
      h_frontier[i] = 1;
      h_values[i] = i;
    }
    break;
  }
  return;
}

template <class EdgeType> void CSR<EdgeType>::SetPartitionsConfig() {

  *numPartitions = (uint32)std::ceil((double)numEdges / maxEdgesInPartition);

  h_partitionCost = new float[*numPartitions];

  std::cout << "Number of edges: " << numEdges << std::endl;
  std::cout << "Maximum number of edges per partition: " << maxEdgesInPartition
            << std::endl;
  std::cout << "Number of partitions: " << *numPartitions << std::endl;

  cudaHostAlloc((void **)&h_partition, *numPartitions * sizeof(*h_partition),
                cudaHostAllocDefault);

  for (uint32 i = 0; i < *numPartitions; i++)
    h_partition[i] = i;

  h_partitionsOffsets = new uint32[*numPartitions + 1];
  h_partitionsOffsets[0] = 0;
  h_partitionsOffsets[*numPartitions] = *numVertices;

  uint32 currentPartition = 0;
  uint32 edgesInCurrentPartition = 0;

  for (uint32 i = 0; i < *numVertices; i++) {
    if ((h_offsets[i + 1] - h_offsets[i]) + edgesInCurrentPartition <=
        maxEdgesInPartition)
      edgesInCurrentPartition =
          edgesInCurrentPartition + (h_offsets[i + 1] - h_offsets[i]);
    else {
      h_partitionsOffsets[++currentPartition] = i;
      edgesInCurrentPartition =
          h_offsets[i + 1] - h_offsets[i]; // This vertex should already be
                                           // included in the new partition

      if (edgesInCurrentPartition > maxEdgesInPartition)
        std::cout << "We've got a problem!" << std::endl;
    }
  }

  return;
}

template <class EdgeType> void CSR<EdgeType>::SetNumStaticEdges() {
  size_t freeMemory;  // Available GPU memory
  size_t totalMemory; // Total GPU memory
  cudaMemGetInfo(&freeMemory,
                 &totalMemory); // Available and Total memory in bytes

  if (algorithm == SSSP)
    freeMemory *= 0.25; // SSSP requires two more allocs of equal size
  else
    freeMemory *= 0.5;

  unsigned long remainder = freeMemory % FRAGMENT;
  freeMemory = freeMemory - remainder; // Free memory in 16kB chunks

  numStaticEdges =
      freeMemory / sizeof(EdgeType); // Available memory in Edges count

  if (numStaticEdges > numEdges) // Potential overflow
    numStaticEdges = numEdges;

  uint64 edgesInStatic = 0;

  numStaticEdges = EDGES_IN_PARTITION;
  for (uint32 i = 0; i < *numVertices; i++) {
    // Build static inStatic array
    if (h_offsets[i + 1] <= numStaticEdges) {
      h_inStatic[i] = true;
      edgesInStatic = edgesInStatic + (h_offsets[i + 1] - h_offsets[i]);
    } else
      h_inStatic[i] = false;
  }

  numStaticEdges = edgesInStatic;

  std::cout << "Number of static edges: " << numStaticEdges << std::endl;

  return;
}

template <class EdgeType> void CSR<EdgeType>::InitData(uint64 sourceVertex) {
  TimeRecord<chrono::milliseconds> process("InitData execution");
  process.startRecord();

  srcVertex = sourceVertex;

  if (algorithm == PR)
    h_valuesPR = new double[*numVertices];
  else
    h_values = new EdgeType[*numVertices];

  h_frontier = new bool[*numVertices];
  // h_filterFrontier = new bool[*numVertices];
  h_inStatic = new bool[*numVertices];

  // Sizes that might get transferred to the GPU (Change to CudaDefault if we
  // end up transferring via cudaMemcpy())
  GPUAssert(cudaHostAlloc((void **)&frontierSize, sizeof(*frontierSize),
                          cudaHostAllocMapped));
  GPUAssert(cudaHostAlloc((void **)&staticSize, sizeof(*staticSize),
                          cudaHostAllocMapped));
  GPUAssert(cudaHostAlloc((void **)&demandSize, sizeof(*demandSize),
                          cudaHostAllocMapped));
  GPUAssert(cudaHostAlloc((void **)&numPartitions, sizeof(*numPartitions),
                          cudaHostAllocMapped));

  GPUAssert(cudaHostAlloc((void **)&h_finished, sizeof(*h_finished),
                          cudaHostAllocDefault));

  GPUAssert(cudaMalloc(&d_finished, sizeof(*d_finished)));

  GPUAssert(cudaHostAlloc((void **)&h_nCoalescedPartitions,
                          sizeof(*h_nCoalescedPartitions),
                          cudaHostAllocDefault));

  GPUAssert(
      cudaMalloc(&d_nCoalescedPartitions, sizeof(*d_nCoalescedPartitions)));

  GPUAssert(cudaHostAlloc((void **)&h_filterOffset,
                          N_FILTER_STREAMS * sizeof(*h_filterOffset),
                          cudaHostAllocDefault));

  // GPUAssert(cudaMalloc(&d_filterOffset, sizeof(*d_filterOffset)));

  GPUAssert(
      cudaMalloc(&d_filterOffset, N_FILTER_STREAMS * sizeof(*d_filterOffset)));

  // *h_finished = 0;
  *frontierSize = 0;
  *staticSize = 0;
  *demandSize = 0;
  *numPartitions = 0;

  InitFrontierNValues();
  // Offsets Array
  GPUAssert(cudaMalloc(&d_offsets, (*numVertices + 1) * sizeof(*d_offsets)));
  GPUAssert(cudaMemcpy(d_offsets, h_offsets,
                       (*numVertices + 1) * sizeof(*d_offsets),
                       cudaMemcpyHostToDevice));

  // Demand List Array
  GPUAssert(cudaMalloc(&d_demandList, *numVertices * sizeof(*d_demandList)));
  GPUAssert(cudaMemset(d_demandList, 0, *numVertices * sizeof(*d_demandList)));

  // Static List Array
  GPUAssert(cudaMalloc(&d_staticList, *numVertices * sizeof(*d_staticList)));
  GPUAssert(cudaMemset(
      d_staticList, 0,
      *numVertices * sizeof(*d_staticList))); // This can only be as big as the
                                              // static Size tho (to improve)

  // Prefix Sum Array
  GPUAssert(cudaMalloc(&d_prefixSum, *numVertices * sizeof(*d_prefixSum)));
  cudaMemset(d_prefixSum, 0,
             *numVertices * sizeof(*d_prefixSum)); // Not reallyh needed I think

  // Frontier Array
  GPUAssert(cudaMalloc(&d_frontier, *numVertices * sizeof(*d_frontier)));
  GPUAssert(cudaMemcpy(d_frontier, h_frontier,
                       *numVertices * sizeof(*d_frontier),
                       cudaMemcpyHostToDevice));

  // Static Frontier Array
  GPUAssert(
      cudaMalloc(&d_staticFrontier, *numVertices * sizeof(*d_staticFrontier)));
  GPUAssert(cudaMemset(d_staticFrontier, 0,
                       *numVertices * sizeof(*d_staticFrontier)));

  // Demand Frontier Array
  GPUAssert(
      cudaMalloc(&d_demandFrontier, *numVertices * sizeof(*d_demandFrontier)));
  GPUAssert(cudaMemset(d_demandFrontier, 0,
                       *numVertices * sizeof(*d_demandFrontier)));

  GPUAssert(
      cudaMalloc(&d_filterFrontier, *numVertices * sizeof(*d_filterFrontier)));
  GPUAssert(cudaMemset(d_filterFrontier, 0,
                       *numVertices * sizeof(*d_filterFrontier)));

  // Filter Frontier Array
  //  GPUAssert(
  //      cudaMalloc(&d_filterFrontier, *numVertices *
  //      sizeof(*d_filterFrontier)));
  //  GPUAssert(cudaMemset(d_filterFrontier, 0,
  //                       *numVertices * sizeof(*d_filterFrontier)));
  //
  //  GPUAssert(
  //      cudaMalloc(&d_partitionNum, N_FILTER_STREAMS *
  //      sizeof(*d_partitionNum)));
  //

  // Mostly Values Array
  if (algorithm == PR) {
    GPUAssert(cudaMalloc(&d_degree, *numVertices * sizeof(*d_degree)));
    GPUAssert(cudaMemcpy(d_degree, h_degree, *numVertices * sizeof(*d_degree),
                         cudaMemcpyHostToDevice));

    GPUAssert(cudaMalloc(&d_valuesPR, *numVertices * sizeof(*d_valuesPR)));
    GPUAssert(cudaMemcpy(d_valuesPR, h_valuesPR,
                         *numVertices * sizeof(*d_valuesPR),
                         cudaMemcpyHostToDevice));

    GPUAssert(cudaMalloc(&d_sum, *numVertices * sizeof(*d_sum)));
    GPUAssert(cudaMemset(d_sum, 0, *numVertices * sizeof(*d_sum)));

    d_thurstSum = thrust::device_ptr<double>(d_sum);
  } else {
    GPUAssert(cudaMalloc(&d_values, *numVertices * sizeof(*d_values)));
    GPUAssert(cudaMemcpy(d_values, h_values, *numVertices * sizeof(*d_values),
                         cudaMemcpyHostToDevice));
  }

  std::cout << "Pre partitions" << std::endl;
  // Sets number of partitions and fills h_partitionsOffsets
  SetPartitionsConfig();

  GPUAssert(cudaMalloc(&d_partitionsOffsets,
                       (*numPartitions + 1) * sizeof(*d_partitionsOffsets)));
  GPUAssert(cudaMemcpy(d_partitionsOffsets, h_partitionsOffsets,
                       (*numPartitions + 1) * sizeof(*d_partitionsOffsets),
                       cudaMemcpyHostToDevice));

  GPUAssert(
      cudaMalloc(&d_partitionCost, *numPartitions * sizeof(*d_partitionCost)));
  GPUAssert(cudaMemset(d_partitionCost, 0,
                       *numPartitions * sizeof(*d_partitionCost)));

  GPUAssert(cudaHostAlloc((void **)&h_finished, sizeof(*h_finished),
                          cudaHostAllocDefault));

  GPUAssert(cudaMalloc(&d_partitionList,
                       N_FILTER_STREAMS * sizeof(*d_partitionList)));

  cudaSetDevice(1);
  GPUAssert(cudaMalloc(&d_neighborPartitionList,
                       N_FILTER_STREAMS * sizeof(*d_neighborPartitionList)));

  cudaSetDevice(2);
  GPUAssert(cudaMalloc(&d_neighborPartitionList2,
                       N_FILTER_STREAMS * sizeof(*d_neighborPartitionList2)));
  cudaSetDevice(3);
  GPUAssert(cudaMalloc(&d_neighborPartitionList3,
                       N_FILTER_STREAMS * sizeof(*d_neighborPartitionList3)));

  cudaSetDevice(0);

  for (uint32 i = 0; i < N_FILTER_STREAMS; ++i) {
    GPUAssert(cudaMalloc(&d_filterEdges[i],
                         maxEdgesInPartition * sizeof(*d_filterEdges)));

    GPUAssert(cudaMemset(d_filterEdges[i], 0,
                         maxEdgesInPartition * sizeof(*d_filterEdges)));

    cudaSetDevice(1);
    GPUAssert(cudaMalloc(&d_neighborFilterEdges[i],
                         maxEdgesInPartition * sizeof(*d_filterEdges)));

    GPUAssert(cudaMemset(d_neighborFilterEdges[i], 0,
                         maxEdgesInPartition * sizeof(*d_filterEdges)));

    cudaSetDevice(2);
    GPUAssert(cudaMalloc(&d_neighborFilterEdges2[i],
                         maxEdgesInPartition * sizeof(*d_filterEdges)));

    GPUAssert(cudaMemset(d_neighborFilterEdges2[i], 0,
                         maxEdgesInPartition * sizeof(*d_filterEdges)));

    cudaSetDevice(3);
    GPUAssert(cudaMalloc(&d_neighborFilterEdges3[i],
                         maxEdgesInPartition * sizeof(*d_filterEdges)));

    GPUAssert(cudaMemset(d_neighborFilterEdges3[i], 0,
                         maxEdgesInPartition * sizeof(*d_filterEdges)));

    cudaSetDevice(0);
  }

  GPUAssert(cudaHostAlloc((void **)&h_partitionList,
                          N_FILTER_STREAMS * sizeof(*h_partitionList),
                          cudaHostAllocDefault));

  GPUAssert(cudaHostAlloc((void **)&h_neighborPartitionList,
                          N_FILTER_STREAMS * sizeof(*h_neighborPartitionList),
                          cudaHostAllocDefault));

  GPUAssert(cudaHostAlloc((void **)&h_neighborPartitionList2,
                          N_FILTER_STREAMS * sizeof(*h_neighborPartitionList2),
                          cudaHostAllocDefault));

  GPUAssert(cudaHostAlloc((void **)&h_neighborPartitionList3,
                          N_FILTER_STREAMS * sizeof(*h_neighborPartitionList3),
                          cudaHostAllocDefault));

  if (algorithm == SSSP)
    GPUAssert(cudaMalloc(&d_filterWeights,
                         maxEdgesInPartition * sizeof(*d_filterWeights)));

  // std::cout << "Post partitions stuff" << std::endl;

  // inStatic Array
  GPUAssert(cudaMalloc(&d_inStatic, *numVertices * sizeof(*d_inStatic)));

  SetNumStaticEdges();

  // inStatic Array
  GPUAssert(cudaMemcpy(d_inStatic, h_inStatic,
                       *numVertices * sizeof(*d_inStatic),
                       cudaMemcpyHostToDevice));

  // Static Edges Array
  GPUAssert(
      cudaMalloc(&d_staticEdges, numStaticEdges * sizeof(*d_staticEdges)));

  GPUAssert(cudaMemcpy(d_staticEdges, h_edges,
                       numStaticEdges * sizeof(*d_staticEdges),
                       cudaMemcpyHostToDevice));

  // Static Weights Array for SSSP
  if (algorithm == SSSP) {
    GPUAssert(cudaMalloc(&d_staticWeights,
                         numStaticEdges * sizeof(*d_staticWeights)));

    GPUAssert(cudaMemcpy(d_staticWeights, h_weights,
                         numStaticEdges * sizeof(*d_staticWeights),
                         cudaMemcpyHostToDevice));
  }

  // Thurst Wrappers
  thrustFrontier = thrust::device_ptr<bool>(d_frontier);
  thurstStaticFrontier = thrust::device_ptr<bool>(d_staticFrontier);
  thurstDemandFrontier = thrust::device_ptr<bool>(d_demandFrontier);
  thurstPrefixSum = thrust::device_ptr<EdgeType>(d_prefixSum);

  GPUAssert(cudaPeekAtLastError());

  process.endRecord();
  process.print();

  return;
}

// Make this robust to errors reading from the file
template <class EdgeType>
void CSR<EdgeType>::ReadInputFile(const std::string &filePath,
                                  ALGORITHM_TYPE algo) {
  TimeRecord<chrono::milliseconds> process("ReadFile execution");
  process.startRecord();

  algorithm = algo;

  cudaHostAlloc((void **)&numVertices, sizeof(*numVertices),
                cudaHostAllocMapped);

  uint64 nV;

  std::ifstream infile(filePath, std::ios::in | std::ios::binary);

  if (!infile.is_open()) {
    std::cerr << " Error opening file to read" << std::endl;
    exit(0);
  }

  infile.read((char *)&nV, sizeof(uint64));
  infile.read((char *)&numEdges, sizeof(uint64));

  *numVertices = static_cast<EdgeType>(nV);
  // numEdges = static_cast<uint64>(nE);

  std::cout << "Num Vertices: " << *numVertices << std::endl;
  std::cout << "Num Edges: " << numEdges << std::endl;

  // uint64 *tempOffsets = new uint64[*numVertices];
  // infile.read((char *)tempOffsets, (*numVertices) * sizeof(uint64));

  // cudaHostAlloc((void **)&h_offsets, (*numVertices + 1) * sizeof(uint64),
  // cudaHostAllocDefault); // just pinned

  h_offsets = new uint64[*numVertices + 1];
  infile.read((char *)h_offsets, (*numVertices) * sizeof(uint64));

  // for (int i = 0; i < *numVertices; ++i)
  //     h_offsets[i] = static_cast<uint64>(tempOffsets[i]);

  h_offsets[*numVertices] = numEdges;

  // delete[] tempOffsets;

  uint64 *tempEdges = new uint64[numEdges];
  infile.read((char *)tempEdges, numEdges * sizeof(uint64));

  // cudaHostAlloc((void **)&h_edges, numEdges * sizeof(EdgeType),
  //              cudaHostAllocMapped);

  h_edges = (EdgeType *)numa_alloc_onnode(numEdges * sizeof(uint32), 0);

  cudaHostRegister(h_edges, numEdges * sizeof(EdgeType),
                   cudaHostRegisterMapped);

  // We can prob parallelize this
  for (int i = 0; i < numEdges; ++i)
    h_edges[i] = static_cast<EdgeType>(tempEdges[i]);

  delete[] tempEdges;

  if (algorithm == SSSP) {
    uint64 *tempWeights = new uint64[numEdges];
    infile.read((char *)tempWeights, numEdges * sizeof(uint64));

    cudaHostAlloc((void **)&h_weights, numEdges * sizeof(*h_weights),
                  cudaHostAllocMapped);

    // We can prob parallelize this
    for (int i = 0; i < numEdges; ++i)
      h_weights[i] = static_cast<EdgeType>(tempWeights[i]);

    delete[] tempWeights;
  } else if (algorithm == PR) {
    uint64 *tempOutDegree = new uint64[*numVertices];
    infile.read((char *)tempOutDegree, *numVertices * sizeof(uint64));

    h_degree = new uint32[*numVertices];

    // We can prob parallelize this

    for (int i = 0; i < *numVertices; ++i)
      h_degree[i] = static_cast<EdgeType>(tempOutDegree[i]);

    delete[] tempOutDegree;
  }

  infile.close();

  // // Debugging
  // for (uint64 i = *numVertices; i > *numVertices - 3; i--)
  //     std::cout << "h_offsets[" << i << "] = " << h_offsets[i] << std::endl;

  // // Debugging
  // for (uint64 i = numEdges - 1; i > numEdges - 3; i--)
  //     std::cout << "h_edges[" << i << "] = " << h_edges[i] << std::endl;

  process.endRecord();
  process.print();
  return;
}

template <class EdgeType> void CSR<EdgeType>::DumpValues() {
  std::string filepath = "results/values1.bin";

  std::ofstream file(filepath);
  if (!file.is_open())
    return;

  // Write values
  for (EdgeType i = 0; i < *numVertices; ++i)
    file.write(reinterpret_cast<const char *>(&h_values[i]), sizeof(EdgeType));

  file.close();
}

template <class EdgeType> void CSR<EdgeType>::Free() {

  if (algorithm == SSSP)
    cudaFreeHost(h_weights);

  else if (algorithm == PR)
    delete[] h_degree;

  delete[] h_offsets;
  cudaFreeHost(h_edges);

  if (algorithm == PR)
    delete[] h_valuesPR;
  else
    delete[] h_values;

  delete[] h_frontier;
  delete[] h_inStatic;

  cudaFreeHost(frontierSize);
  cudaFreeHost(staticSize);
  cudaFreeHost(demandSize);
  cudaFreeHost(numPartitions);

  // cudaFreeHost(h_nCoalescedPartitions);

  // cudaFreeHost(h_finished);

  // cudaFree(d_finished);
  // cudaFree(d_nCoalescedPartitions);

  cudaFree(d_offsets);

  cudaFree(d_demandList);

  cudaFree(d_staticList);

  cudaFree(d_prefixSum);

  cudaFree(d_frontier);

  cudaFree(d_staticFrontier);

  cudaFree(d_demandFrontier);

  // cudaFree(d_filterFrontier);

  if (algorithm == PR) {
    cudaFree(d_degree);
    cudaFree(d_valuesPR);
    cudaFree(d_sum);
  } else
    cudaFree(d_values);

  // delete[] h_filterPartitionCost;
  delete[] h_partitionsOffsets;

  // delete[] h_partitionCost;

  cudaFree(d_partitionsOffsets);
  // cudaFree(d_filterPartitionCost);
  cudaFree(d_partitionCost);

  cudaFree(d_filterEdges);

  cudaFree(d_inStatic);
  cudaFree(d_staticEdges);

  if (algorithm == SSSP) {
    //  cudaFree(d_filterWeights);
    cudaFree(d_staticWeights);
  }

  GPUAssert(cudaPeekAtLastError());

  return;
}

#endif // GRAPH_CUH

// Antigo Reset
// switch (algorithm)
// {
// case PR:
//     for (uint32 i = 0; i < *numVertices; i++)
//     {
//         h_frontier[i] = 1;
//         h_valuesPR[i] = 1.0 / *numVertices;
//     }
//     cudaMemcpy(d_valuesPR, h_valuesPR, *numVertices * sizeof(double),
//     cudaMemcpyHostToDevice); break;

// case BFS:
//     for (uint32 i = 0; i < *numVertices; i++)
//     {
//         h_frontier[i] = 0;
//         h_values[i] = *numVertices + 1;
//     }
//     h_frontier[srcVertex] = 1;
//     h_values[srcVertex] = 1;
//     cudaMemcpy(d_values, h_values, *numVertices * sizeof(EdgeType),
//     cudaMemcpyHostToDevice); break;

// case CC:
//     for (uint32 i = 0; i < *numVertices; i++)
//     {
//         h_frontier[i] = 1;
//         h_values[i] = i;
//     }
//     cudaMemcpy(d_values, h_values, *numVertices * sizeof(EdgeType),
//     cudaMemcpyHostToDevice); break;

// case SSSP:
//     for (uint32 i = 0; i < *numVertices; i++)
//     {
//         h_frontier[i] = 0;
//         h_values[i] = *numVertices + 1;
//     }
//     h_frontier[srcVertex] = 1;
//     h_values[srcVertex] = 1;
//     cudaMemcpy(d_values, h_values, *numVertices * sizeof(EdgeType),
//     cudaMemcpyHostToDevice); break;
// }
