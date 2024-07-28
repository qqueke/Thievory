#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// #define WEIGHTED 0
#define UNDIRECTED
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

void writeToFile(const std::string &filename, uint64 numElements,
                 const void *data, size_t elementSize) {
  std::ofstream outFile(filename, std::ios::binary);

  if (!outFile) {
    std::cerr << "Error opening file: " << filename << std::endl;
    return;
  }

  // Write the header
  outFile.write(reinterpret_cast<const char *>(&numElements), sizeof(uint64));
  uint64 placeholder = 0;
  outFile.write(reinterpret_cast<const char *>(&placeholder), sizeof(uint64));

  // Write the data
  outFile.write(reinterpret_cast<const char *>(data),
                numElements * elementSize);

  outFile.close();
}

std::string getOutputFilePath(const std::string &basePath,
                              const std::string &extension) {
  size_t lastDot = basePath.find_last_of(".");
  std::string newPath = basePath.substr(0, lastDot) + extension;
  return newPath;
}

void ConvertTxtToHWEL(std::string filePath, uint32 linesToSkip) {
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
  uint64 src = 0, dest = 0, weight = 20;

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

  // Sort according to destiny
  std::sort(edgesVector.begin(), edgesVector.end(),
            [](const Edge &a, const Edge &b) {
              if (a.dest == b.dest)
                return a.src < b.src;
              else
                return a.dest < b.dest;
            });

  std::ofstream outfile(filePath.substr(0, filePath.length() - 3) + "el");

  if (!outfile.is_open()) {
    std::cerr << "Failed to open file to write " << std::endl;
    return;
  }

  for (const auto &edge : edgesVector) {
    outfile << edge.src << "\t" << edge.dest << "\t" << edge.weight << "\n";
  }

  outfile.close();
  if (!outfile)
    std::cerr << "Error occurred when writing to file: " << std::endl;

  return;
}

void ConvertTxtToWEL(std::string filePath, uint32 linesToSkip) {
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
  uint64 src = 0, dest = 0, weight = 20;

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

  // Sort according to destiny
  std::sort(edgesVector.begin(), edgesVector.end(),
            [](const Edge &a, const Edge &b) {
              if (a.dest == b.dest)
                return a.src < b.src;
              else
                return a.dest < b.dest;
            });

  std::ofstream outfile(filePath.substr(0, filePath.length() - 3) + "el");

  if (!outfile.is_open()) {
    std::cerr << "Failed to open file to write " << std::endl;
    return;
  }

  for (const auto &edge : edgesVector) {
    outfile << edge.src << " " << edge.dest << "\n";
  }

  outfile.close();
  if (!outfile)
    std::cerr << "Error occurred when writing to file: " << std::endl;

  return;
}

void ConvertTxtToEL(std::string filePath, uint32 linesToSkip) {
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
  uint64 src = 0, dest = 0, weight = 20;

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

  // Sort according to destiny
  std::sort(edgesVector.begin(), edgesVector.end(),
            [](const Edge &a, const Edge &b) {
              if (a.dest == b.dest)
                return a.src < b.src;
              else
                return a.dest < b.dest;
            });

  std::ofstream outfile(filePath.substr(0, filePath.length() - 3) + "el");

  if (!outfile.is_open()) {
    std::cerr << "Failed to open file to write " << std::endl;
    return;
  }

  for (const auto &edge : edgesVector) {
    outfile << edge.src << " " << edge.dest << "\n";
  }

  outfile.close();
  if (!outfile)
    std::cerr << "Error occurred when writing to file: " << std::endl;

  return;
}

void ConvertTxtToBCSC(std::string filePath, uint32 linesToSkip) {
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
  uint64 src = 0, dest = 0, weight = 20;

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

  std::vector<uint32> inDegree(numVertices, 0);
  for (uint64 i = 0; i < edgesVector.size(); ++i)
    inDegree[edgesVector[i].dest]++;

  std::vector<uint32> outDegree(numVertices, 0);
  for (uint64 i = 0; i < edgesVector.size(); ++i)
    outDegree[edgesVector[i].src]++;

  std::vector<uint64> offsets(numVertices, 0);
  std::vector<uint32> edges(numEdges, 0);

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

  for (uint64 i = 0; i < numEdges; i++) {
    edges[i] = (uint32)edgesVector[i].src;
  }

  // Hard coded to extract the .txt and put it as bcsr (Byte CSR)
  std::ofstream outfile(filePath.substr(0, filePath.length() - 3) + "csc",
                        std::ofstream::binary);

  if (!outfile.is_open()) {
    std::cerr << "Error opening file for writing." << std::endl;
    exit(0);
  }

  // Header
  outfile.write((char *)&numVertices, sizeof(uint64));
  outfile.write((char *)&numEdges, sizeof(uint64));

  // Data
  outfile.write(reinterpret_cast<const char *>(offsets.data()),
                offsets.size() * sizeof(uint64));

  outfile.write(reinterpret_cast<const char *>(edges.data()),
                edges.size() * sizeof(uint32));

  outfile.write(reinterpret_cast<const char *>(outDegree.data()),
                outDegree.size() * sizeof(uint32));

  outfile.close();

  return;
}
void ConvertTxtToEMOGI(std::string filePath, uint32 linesToSkip) {
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
  uint64 src = 0, dest = 0, weight = 20;

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

  std::vector<uint64> degree(numVertices, 0);

  for (uint64 i = 0; i < edgesVector.size(); ++i)
    degree[edgesVector[i].src]++;

  std::vector<uint64> offsets(numVertices + 1, 0);
  std::vector<uint64> edges(numEdges, 0);
  std::vector<uint32> weights(numEdges, 0);

  uint64 temp = 0;
  for (uint64 i = 0; i < numVertices; i++) {
    offsets[i] = temp;
    temp += degree[i];
  }

  offsets[numVertices] = numEdges;

  std::sort(edgesVector.begin(), edgesVector.end());

  for (uint64 i = 0; i < edgesVector.size(); i++) {
    edges[i] = (uint64)edgesVector[i].dest;
    weights[i] = (uint32)edgesVector[i].weight;
  }

  std::string colFilePath = getOutputFilePath(filePath, ".bel.col");
  std::string dstFilePath = getOutputFilePath(filePath, ".bel.dst");
  std::string valFilePath = getOutputFilePath(filePath, ".bel.val");

  writeToFile(colFilePath, numVertices, offsets.data(), sizeof(uint64));
  writeToFile(dstFilePath, numEdges, edges.data(), sizeof(uint64));
  writeToFile(valFilePath, numEdges, weights.data(), sizeof(uint32));

  return;
}

void ConvertTxtToBCSR(std::string filePath, uint32 linesToSkip,
                      bool liberator) {
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
  uint64 src = 0, dest = 0, weight = 20;

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

  std::vector<uint64> degree(numVertices, 0);

  for (uint64 i = 0; i < edgesVector.size(); ++i)
    degree[edgesVector[i].src]++;

  std::vector<uint64> offsets(numVertices, 0);
  std::vector<uint32> edges(numEdges, 0);
  std::vector<uint32> weights(numEdges, 0);

  uint64 temp = 0;
  for (uint64 i = 0; i < numVertices; i++) {
    offsets[i] = temp;
    temp += degree[i];
  }

  std::sort(edgesVector.begin(), edgesVector.end());

  for (uint64 i = 0; i < edgesVector.size(); i++) {
    edges[i] = (uint32)edgesVector[i].dest;
    weights[i] = (uint32)edgesVector[i].weight;
  }

  if (!liberator) {
    // Hard coded to extract the .txt and put it as bcsr (Byte CSR)
    std::ofstream outfile(filePath.substr(0, filePath.length() - 3) + "csr",
                          std::ofstream::binary);

    if (!outfile.is_open()) {
      std::cerr << "Error opening file for writing." << std::endl;
      exit(0);
    }

    // Header
    outfile.write((char *)&numVertices, sizeof(numVertices));
    outfile.write((char *)&numEdges, sizeof(numEdges));

    // Data
    outfile.write(reinterpret_cast<const char *>(offsets.data()),
                  offsets.size() * sizeof(uint64));

    outfile.write(reinterpret_cast<const char *>(edges.data()),
                  edges.size() * sizeof(uint32));

    outfile.write(reinterpret_cast<const char *>(weights.data()),
                  weights.size() * sizeof(uint32));

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
    outfile.write(reinterpret_cast<const char *>(offsets.data()),
                  offsets.size() * sizeof(uint64));

    //  LiberatorEdge *edgesLiberator = new LiberatorEdge[numEdges];

    std::vector<LiberatorEdge> edgesLiberator;

    for (unsigned long long i = 0; i < numEdges; i++) {
      edgesLiberator.push_back({edges[i], weights[i]});
    }

    outfile.write(reinterpret_cast<const char *>(edgesLiberator.data()),
                  edgesLiberator.size() * sizeof(LiberatorEdge));

    outfile.close();
  }

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

  if (type == "csr") {
    std::cout << "Converting to CSR" << std::endl;
    ConvertTxtToBCSR(filePath, linesToSkip, false);
  } else if (type == "csc") {
    std::cout << "Converting to CSC" << std::endl;
    ConvertTxtToBCSC(filePath, linesToSkip);
  } else if (type == "wcsr") {
    std::cout << "Converting to WCSR" << std::endl;
    ConvertTxtToBCSR(filePath, linesToSkip, true);
  } else if (type == "emogi") {
    std::cout << "Converting to EMOGI" << std::endl;
    ConvertTxtToEMOGI(filePath, linesToSkip);
  } else if (type == "subway") {
    std::cout << "Converting to Subway" << std::endl;
    ConvertTxtToEL(filePath, linesToSkip);
  } else if (type == "subwayWeights") {
    std::cout << "Converting to Subway with Weights" << std::endl;
    ConvertTxtToWEL(filePath, linesToSkip);
  } else if (type == "hytgraph") {
    std::cout << "Converting to HyTGraph" << std::endl;
    ConvertTxtToHWEL(filePath, linesToSkip);
  } else
    exit(0);
  return 0;
}
