#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// #define WEIGHTED 0
// #define UNDIRECTED
typedef unsigned int uint32;
typedef unsigned long long uint64;

struct LiberatorEdge {
  uint32 dest;
  uint32 weight;
  operator uint32() const { return (dest << 16) | weight; }
};

struct Edge {
  uint64 src;
  uint64 dest;
  uint64 weight;

  Edge() : src(0), dest(0), weight(0) {}
  Edge(uint64 s, uint64 d) : src(s), dest(d), weight(1) {}
  Edge(uint64 s, uint64 d, uint64 w) : src(s), dest(d), weight(w) {}

  bool operator<(const Edge &other) const {
    if (src == other.src)
      return dest < other.dest;
    else
      return src < other.src;
  }
};

void ConvertTxtToBCSC(std::string filePath, uint32 linesToSkip) {
  std::cout << "Converting to CSC" << std::endl;
  std::string line;

  std::ifstream infile;
  infile.open(filePath);

  if (!infile.is_open()) {
    std::cerr << "Error opening file for reading." << std::endl;
    exit(0);
  }

  for (uint32 i = 0; i < linesToSkip; ++i)
    getline(infile, line);

  uint64 numVertices = 0, numEdges = 0;
  uint64 src = 0, dest = 0, weight = 0;

  std::vector<Edge> edgesVector;

  std::stringstream ss;

  while (getline(infile, line)) {
    ss.str("");
    ss.clear();
    ss << line;
    ss >> src;
    ss >> dest;

#ifdef WEIGHTED
    ss >> weight;
#endif
    edgesVector.emplace_back(src, dest, weight);

#ifdef UNDIRECTED
    edgesVector.emplace_back(dest, src, weight);
#endif

    // Acquire the number of vertices
    if (numVertices < src)
      numVertices = src;

    if (numVertices < dest)
      numVertices = dest;
  }

  infile.close();

  numVertices++; // 0 is included
  numEdges = edgesVector.size();

  uint64 *inDegree = (uint64 *)calloc(numVertices, sizeof(*inDegree));

  for (uint64 i = 0; i < edgesVector.size(); ++i)
    inDegree[edgesVector[i].dest]++;

  uint64 *outDegree = (uint64 *)calloc(numVertices, sizeof(*outDegree));

  for (uint64 i = 0; i < edgesVector.size(); ++i)
    outDegree[edgesVector[i].src]++;

  uint64 *offsets = (uint64 *)calloc(numVertices, sizeof(*offsets));
  uint64 *edges = (uint64 *)calloc(numEdges, sizeof(*edges));
  // uint64 *weights = (uint64 *)calloc(numEdges, sizeof(*weights));

  uint64 temp = 0;
  for (uint64 i = 0; i < numVertices; i++) {
    offsets[i] = temp;
    temp += inDegree[i];
  }

  // Sort according to destiny
  std::sort(edgesVector.begin(), edgesVector.end(),
            [](const Edge &a, const Edge &b) {
              if (a.dest == b.dest)
                return a.src < b.src;
              else
                return a.dest < b.dest;
            });

  for (uint64 i = 0; i < edgesVector.size(); ++i) {
    edges[i] = edgesVector[i].src;

    // #ifdef WEIGHTED
    //         weights[i] = edgesVector[i].weight;
    // #else
    //         weights[i] = 20;
    // #endif
  }

  // Hard coded to extract the .txt and put it as bcsr (Byte CSR)
  std::ofstream outfile(filePath.substr(0, filePath.length() - 3) + "csc",
                        std::ofstream::binary);

  if (!outfile.is_open()) {
    std::cerr << "Error opening file for writing." << std::endl;
    exit(0);
  }

  // First the header (numVertices and numEdges)
  outfile.write((char *)&numVertices, sizeof(numVertices));
  outfile.write((char *)&numEdges, sizeof(numEdges));
  // Offsets, Edges, OutDegree (uint64)
  outfile.write((char *)offsets, numVertices * sizeof(*offsets));
  outfile.write((char *)edges, numEdges * sizeof(*edges));
  outfile.write((char *)outDegree, numVertices * sizeof(*outDegree));

  outfile.close();

  return;
}

void ConvertTxtToBCSR(std::string filePath, uint32 linesToSkip,
                      bool liberator) {
  std::cout << "Converting to CSR" << std::endl;

  std::string line;

  std::ifstream infile;
  infile.open(filePath);

  if (!infile.is_open()) {
    std::cerr << "Error opening file for reading: " << strerror(errno)
              << std::endl;
    exit(EXIT_FAILURE);
  }

  for (uint32 i = 0; i < linesToSkip; ++i)
    getline(infile, line);

  uint64 numVertices = 0, numEdges = 0;
  uint64 src = 0, dest = 0, weight = 0;

  std::vector<Edge> edgesVector;

  std::stringstream ss;

  while (getline(infile, line)) {
    ss.str("");
    ss.clear();
    ss << line;
    ss >> src;
    ss >> dest;

#ifdef WEIGHTED
    ss >> weight;
#endif
    edgesVector.emplace_back(src, dest, weight);

#ifdef UNDIRECTED
    edgesVector.emplace_back(dest, src, weight);
#endif

    // Acquire the number of vertices
    if (numVertices < src)
      numVertices = src;

    if (numVertices < dest)
      numVertices = dest;
  }

  infile.close();

  numVertices++; // 0 is included
  numEdges = edgesVector.size();

  uint64 *degree = (uint64 *)calloc(numVertices, sizeof(*degree));

  for (uint64 i = 0; i < edgesVector.size(); ++i)
    degree[edgesVector[i].src]++;

  uint64 *offsets = (uint64 *)calloc(numVertices, sizeof(*offsets));
  uint64 *edges = (uint64 *)calloc(numEdges, sizeof(*edges));
  uint64 *weights = (uint64 *)calloc(numEdges, sizeof(*weights));

  uint64 temp = 0;
  for (uint64 i = 0; i < numVertices; i++) {
    offsets[i] = temp;
    temp += degree[i];
  }

  std::sort(edgesVector.begin(), edgesVector.end());

  for (uint64 i = 0; i < edgesVector.size(); ++i) {
    edges[i] = edgesVector[i].dest;

#ifdef WEIGHTED
    weights[i] = edgesVector[i].weight;
#else
    weights[i] = 20;
#endif
  }

  if (!liberator) {
    // Hard coded to extract the .txt and put it as bcsr (Byte CSR)
    std::ofstream outfile(filePath.substr(0, filePath.length() - 3) + ".csr",
                          std::ofstream::binary);

    if (!outfile.is_open()) {
      std::cerr << "Error opening file for writing." << std::endl;
      exit(0);
    }

    // First the header (numVertices and numEdges)
    outfile.write((char *)&numVertices, sizeof(numVertices));
    outfile.write((char *)&numEdges, sizeof(numEdges));

    // Then the offsets array
    outfile.write((char *)offsets, numVertices * sizeof(*offsets));

    // Then the edges array
    outfile.write((char *)edges, numEdges * sizeof(*edges));

    // Finally the weights
    outfile.write((char *)weights, numEdges * sizeof(*weights));

    outfile.close();
  } else {
    // Hard coded to extract the .txt and put it as bcsr (Byte CSR)
    std::ofstream outfile(filePath.substr(0, filePath.length() - 3) + ".wcsr",
                          std::ofstream::binary);

    if (!outfile.is_open()) {
      std::cerr << "Error opening file for writing." << std::endl;
      exit(0);
    }

    // First the header (numVertices and numEdges)
    outfile.write((char *)&numVertices, sizeof(numVertices));
    outfile.write((char *)&numEdges, sizeof(numEdges));

    // Then the offsets array
    outfile.write((char *)offsets, numVertices * sizeof(*offsets));

    //  LiberatorEdge *edgesLiberator = new LiberatorEdge[numEdges];

    std::vector<LiberatorEdge> edgesLiberator;

    for (unsigned long long i = 0; i < numEdges; i++) {
      edgesLiberator.push_back({(uint32)edges[i], 20});
    }

    outfile.write(reinterpret_cast<const char *>(edgesLiberator.data()),
                  edgesLiberator.size() * sizeof(LiberatorEdge));

    outfile.close();
  }

  return;
}

// Liberator specific for weights (SSSP)
void ConvertTxtToBWCSR(std::string filePath, uint32 linesToSkip) {
  std::cout << "Converting to WCSR" << std::endl;

  std::string line;

  std::ifstream infile;
  infile.open(filePath);

  if (!infile.is_open()) {
    std::cerr << "Error opening file for reading." << std::endl;
    exit(0);
  }

  for (uint32 i = 0; i < linesToSkip; ++i)
    getline(infile, line);

  uint64 numVertices = 0, numEdges = 0;
  uint64 src = 0, dest = 0, weight = 0;

  std::vector<Edge> edgesVector;

  std::stringstream ss;

  while (getline(infile, line)) {
    ss.str("");
    ss.clear();
    ss << line;
    ss >> src;
    ss >> dest;

#ifdef WEIGHTED
    ss >> weight;
#endif
    edgesVector.emplace_back(src, dest, weight);

#ifdef UNDIRECTED
    edgesVector.emplace_back(dest, src, weight);
#endif

    // Acquire the number of vertices
    if (numVertices < src)
      numVertices = src;

    if (numVertices < dest)
      numVertices = dest;
  }

  infile.close();

  numVertices++; // 0 is included
  numEdges = edgesVector.size();

  uint64 *degree = (uint64 *)calloc(numVertices, sizeof(*degree));

  for (uint64 i = 0; i < edgesVector.size(); ++i)
    degree[edgesVector[i].src]++;

  LiberatorEdge *edges = new LiberatorEdge[numEdges];
  uint64 *offsets = (uint64 *)calloc(numVertices, sizeof(*offsets));

  uint64 temp = 0;
  for (uint64 i = 0; i < numVertices; i++) {
    offsets[i] = temp;
    temp += degree[i];
  }

  std::sort(edgesVector.begin(), edgesVector.end());

  for (uint64 i = 0; i < edgesVector.size(); ++i) {
    edges[i].dest = edgesVector[i].dest;

#ifdef WEIGHTED
    edges[i].weight[i] = edgesVector[i].weight;
#else
    edges[i].weight = 20;
#endif
  }

  // Hard coded to extract the .txt and put it as bcsr (Byte CSR)
  std::ofstream outfile(filePath.substr(0, filePath.length() - 3) + "wcsr",
                        std::ofstream::binary);

  if (!outfile.is_open()) {
    std::cerr << "Error opening file for writing." << std::endl;
    exit(0);
  }

  // First the header (numVertices and numEdges)
  outfile.write((char *)&numVertices, sizeof(numVertices));
  outfile.write((char *)&numEdges, sizeof(numEdges));

  // Then the offsets array
  outfile.write((char *)offsets, numVertices * sizeof(*offsets));

  // Then the edges array
  outfile.write((char *)edges, numEdges * sizeof(*edges));

  outfile.close();

  return;
}

int main(int argc, char **argv) {
  // Default parameters
  std::string filePath;
  std::string type = "csr";
  bool hasInput = false;
  uint32 linesToSkip = 4;
  uint32 weighted = 0;

  try {
    for (unsigned int i = 1; i < argc - 1; i = i + 2) {
      if (strcmp(argv[i], "--input") == 0) {
        filePath = std::string(argv[i + 1]);
        hasInput = true;
      } else if (strcmp(argv[i], "--skip") == 0)
        linesToSkip = atoi(argv[i + 1]);

      else if (strcmp(argv[i], "--type") == 0)
        type = std::string(argv[i + 1]);
    }
  } catch (...) {
    std::cerr << "An exception has occurred." << std::endl;
    exit(0);
  }

  if (type == "csr")
    ConvertTxtToBCSR(filePath, linesToSkip, false);
  else if (type == "csc")
    ConvertTxtToBCSC(filePath, linesToSkip);
  else if (type == "wcsr")
    ConvertTxtToBCSR(filePath, linesToSkip, true);
  else
    exit(0);
  return 0;
}
