# Thievory

This framework is designed for efficient processing of large graphs for BFS, SSSP, CC, and PRvarious algorithms.

## Table of Contents
- [Installation](#installation)
- [Graph Conversion](#graph-conversion)
- [Usage](#usage)
- [Options](#options)
- [Example](#example)
- [License](#license)

## Installation

### Prerequisites

In order to compile the code you must install:

- A C++ compiler
- NVIDIA CUDA toolkit
- NUMA (Non-Uniform Memory Access) library

### Setup

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/qqueke/Thievory.git
   cd Thievory```

2. Update Makefile

   Replace -arch_sm=80 with your GPU architecture
   Replace $(HOME)/local/include and $(HOME)/local/lib with the appropriate paths to your NUMA installation

3. Compile the code
  ```bash
  make```


   






   
