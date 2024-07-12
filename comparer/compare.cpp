#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>

typedef unsigned int uint32;
typedef unsigned long long uint64;

enum BYTES
{
    _4BYTE = 32,
    _8BYTE = 64,
};

void usage(const char *program_name)
{
    std::cout << "Usage: " << program_name << " <inputFile> <numVertices> <n_ignored_lines> [-d] [-w] [-p]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --input : Specify the graph input " << std::endl;
    std::cout << "  --algo : Specify the algorithm {'bfs', 'cc', 'pr', 'sssp'} (default = 'bfs')" << std::endl;
    std::cout << "  --type : Specify the edge size {32, 64} (default = 32)" << std::endl;
    std::cout << "  --type : Specify the source vertex {0, 1, 2, ...} (default = 1)" << std::endl;
    std::cout << "  --runs : Specify the heuristic for memory usage [0.0f - 1.0] (default = 0.5f)" << std::endl;
    std::cout << "  --runs : Specify the number of runs {0, 1, 2, ...} (default = 1)" << std::endl;
    exit(0);
}

int main(int argc, char **argv)
{

    // std::string filePath;
    // bool hasInput = false;
    // uint64 numVertices = 0;
    // bool hasVertices = false;

    // std::string type1 = "uint32";
    // std::string type2 = "uint32";

    // uint32 type = _4BYTE;
    // uint32 srcVertex = 1;
    // double memAdvise = 0.5f;
    // uint32 nRuns = 1;

    // try
    // {
    //     for (unsigned int i = 1; i < argc - 1; i = i + 2)
    //     {
    //         if (strcmp(argv[i], "--input") == 0)
    //         {
    //             filePath = std::string(argv[i + 1]);
    //             hasInput = true;
    //         }
    //         else if (strcmp(argv[i], "--vertices") == 0)
    //         {
    //             numVertices = atoi(argv[i + 1]);
    //             hasVertices = true;
    //         }
    //         else if (strcmp(argv[i], "--type1") == 0)
    //             type1 = std::string(argv[i + 1]);

    //         else if (strcmp(argv[i], "--type2") == 0)
    //             type2 = std::string(argv[i + 1]);
    //     }
    // }
    // catch (...)
    // {
    //     std::cerr << "An exception has occurred.\n";
    //     exit(0);
    // }

    // if (!hasInput)
    //     exit(0);

    // Work on this (it is currently hardcoded)

    uint32 numVertices = 3072441; // Assuming this is defined and represents the number of vertices
    std::vector<uint32> h_values(numVertices);

    std::string filepath = "../results/values1.bin";

    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open())
        return 1;

    // Read values from the file
    file.read(reinterpret_cast<char *>(h_values.data()), numVertices * sizeof(uint32));

    // Close the file
    file.close();

    std::vector<uint32> h_values2(numVertices);

    std::string filepath2 = "/home/inesc/joaobrotas/Liberator/cmake-build-debug/results/values1.bin";

    std::ifstream file2(filepath2, std::ios::binary);
    if (!file2.is_open())
        return 1;

    // Read values from the file
    file2.read(reinterpret_cast<char *>(h_values2.data()), numVertices * sizeof(uint32));

    file2.close();

    std::cout << "Our result | Their result " << std::endl;

    int nErrors = 0;
    for (uint32 i = 0; i < numVertices; i++)
    {
        if (h_values[i] != h_values2[i])
        {
            std::cout << "Arrays do not match at: " << i << std::endl;
            std::cout << h_values[i] << " | " << h_values2[i] << std::endl;
            nErrors++;
            if (nErrors > 20)
                break;
        }
    }

    // std::vector<double> h_values(numVertices);

    // std::string filepath = "../results/values1.bin";

    // std::ifstream file(filepath, std::ios::binary);
    // if (!file.is_open())
    //     return 1;

    // // Read values from the file
    // file.read(reinterpret_cast<char *>(h_values.data()), numVertices * sizeof(double));

    // // Close the file
    // file.close();

    // std::vector<double> h_values2(numVertices);

    // std::string filepath2 = "/home/inesc/joaobrotas/Liberator/cmake-build-debug/results/values1.bin";

    // std::ifstream file2(filepath2, std::ios::binary);
    // if (!file2.is_open())
    //     return 1;

    // // Read values from the file
    // file2.read(reinterpret_cast<char *>(h_values2.data()), numVertices * sizeof(double));

    // file2.close();

    // std::cout << "Our result | Their result " << std::endl;

    // double tolerance = 1e-3; // Adjust as needed

    // int nErrors = 0;
    // for (uint32 i = 0; i < numVertices; i++)
    // {

    //     double diff = std::abs(h_values[i] - h_values2[i]);

    //     if (diff > tolerance)
    //     {
    //         std::cout << "Arrays do not match at: " << i << std::endl;
    //         std::cout << h_values[i] << " | " << h_values2[i] << std::endl;
    //         nErrors++;
    //         if (nErrors > 20)
    //             break;
    //     }
    // }

    return 1;
}