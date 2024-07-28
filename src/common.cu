#include "common.cuh"

// void GPUAssert(cudaError_t code) {
// if (code != cudaSuccess) {
//     std::cerr << cudaGetErrorName(code) << __FILE__ << __LINE__ <<
//     std::endl;

// fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), __FILE__,
//       __LINE__);

// exit(code);
// }
//}

template <typename EdgeType>
__global__ void
setStaticNDemandFrontiers(EdgeType *h_numVertices, bool *d_frontier,
                          bool *d_staticFrontier, bool *d_demandFrontier,
                          bool *d_inStatic) {
  for (EdgeType vertexId = blockIdx.x * blockDim.x + threadIdx.x;
       vertexId < *h_numVertices; vertexId += blockDim.x * gridDim.x) {
    if (d_frontier[vertexId]) {
      if (d_inStatic[vertexId])
        d_staticFrontier[vertexId] = 1;
      else
        d_demandFrontier[vertexId] = 1;
    }
  }
}
