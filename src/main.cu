#include "../src/bfs/bfs.cuh"
#include "../src/cc/cc.cuh"
#include "../src/pr/pr.cuh"
#include "../src/sssp/sssp.cuh"
#include <iostream>
#include <numa.h>
enum BYTES {
  _4BYTE = 32,
  _8BYTE = 64,
};

void usage(const char *program_name) {
  cout << "Usage: " << program_name
       << " <inputFile> <numVertices> <n_ignored_lines> [-d] [-w] [-p]" << endl;
  cout << "Options:" << endl;
  cout << "  --input : Specify the graph input " << endl;
  cout << "  --algo : Specify the algorithm {'bfs', 'cc', 'pr', 'sssp'} "
          "(default = 'bfs')"
       << endl;
  cout << "  --type : Specify the edge size {32, 64} (default = 32)" << endl;
  cout << "  --type : Specify the source vertex {0, 1, 2, ...} (default = 1)"
       << endl;
  cout << "  --runs : Specify the heuristic for memory usage [0.0f - 1.0] "
          "(default = 0.5f)"
       << endl;
  cout << "  --runs : Specify the number of runs {0, 1, 2, ...} (default = 1)"
       << endl;
  exit(0);
}

int main(int argc, char **argv) {

  cudaFree(0);
  cudaDeviceProp deviceProperties;
  memset(&deviceProperties, 0, sizeof(cudaDeviceProp));

  deviceProperties.major = 6;
  deviceProperties.minor = 0;

  int device; // Selected device
  cudaChooseDevice(
      &device, &deviceProperties); // Select the device that fits criteria best
  cudaSetDevice(device);
  cudaSetDevice(0);

  numa_run_on_node(0);
  std::cout << "Selected device " << device << std::endl;

  // Default parameters
  string filePath;
  bool hasInput = false;
  string algorithm = "bfs";
  uint32 type = _4BYTE;
  uint32 srcVertex = 1;
  double memAdvise = 0.5f;
  uint32 nRuns = 1;

  try {
    for (unsigned int i = 1; i < argc - 1; i = i + 2) {
      if (strcmp(argv[i], "--input") == 0) {
        filePath = string(argv[i + 1]);
        hasInput = true;
      } else if (strcmp(argv[i], "--algo") == 0)
        algorithm = string(argv[i + 1]);

      else if (strcmp(argv[i], "--type") == 0)
        type = atoi(argv[i + 1]);

      else if (strcmp(argv[i], "--source") == 0)
        srcVertex = atoi(argv[i + 1]);

      else if (strcmp(argv[i], "--adviseK") == 0)
        memAdvise = atof(argv[i + 1]);

      else if (strcmp(argv[i], "--runs") == 0)
        nRuns = atoi(argv[i + 1]);
    }
  } catch (...) {
    std::cerr << "An exception has occurred.\n";
    exit(0);
  }

  if (!hasInput)
    exit(0);

  if (algorithm == "bfs") {
    if (type == _4BYTE)
      BFS32(filePath, srcVertex, memAdvise, nRuns);

    else if (type == _8BYTE)
      BFS64(filePath, srcVertex, memAdvise, nRuns);
    else
      usage(argv[0]);
  } else if (algorithm == "cc") {
    if (type == _4BYTE)
      CC32(filePath, memAdvise, nRuns);

    else if (type == _8BYTE)
      CC64(filePath, memAdvise, nRuns);
    else
      usage(argv[0]);
  } else if (algorithm == "sssp") {
    if (type == _4BYTE)
      SSSP32(filePath, srcVertex, memAdvise, nRuns);
    else if (type == _8BYTE)
      SSSP64(filePath, srcVertex, memAdvise, nRuns);
    else
      usage(argv[0]);
  } else if (algorithm == "pr") {
    if (type == _4BYTE)
      PR32(filePath, memAdvise, nRuns);

    else if (type == _8BYTE)
      PR64(filePath, memAdvise, nRuns);
    else
      usage(argv[0]);
  } else
    usage(argv[0]);

  return 0;
}
