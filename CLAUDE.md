# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Liberator is a GPU-accelerated out-of-core graph processing framework that handles graph datasets exceeding GPU memory. It partitions graph data between static (GPU-resident) and dynamic (CPU-resident, streamed on demand) partitions to minimize GPU-CPU data transfers. Published in IEEE TPDS 2023.

Supports four graph algorithms: BFS, SSSP, Connected Components (CC), and PageRank (PR).

## Build

Requires: CMake 3.17+, g++, CUDA toolkit (originally 11.4; CMakeLists.txt currently targets `sm_120`).

```bash
mkdir -p cmake-build-debug && cd cmake-build-debug
cmake ..
make
```

The executable is `ptGraph` (note: `set_target_properties` in CMakeLists.txt references target `Liberator` but `add_executable` creates `ptGraph` — this is a known inconsistency).

You may need to adjust `CMAKE_CUDA_COMPILER` path and `CUDA_ARCHITECTURES` in CMakeLists.txt for your GPU.

## Run

```bash
./ptGraph \
  --input <graph_file_path> \
  --type <bfs|cc|sssp|pr> \
  --sourceNode <id>       # required for bfs and sssp \
  --model <0|7>           # 0=Ascetic (previous work), 7=Liberator \
  --testTime <n>          # number of iterations \
  --adviseK <ratio>       # controls static/overload partition ratio
```

Input formats: `.bcsr` (BFS/CC), `.bcsc` (PageRank), `.bwcsr` (SSSP). Use `converter/converter.cpp` to convert text edge lists to binary formats.

## Architecture

### Entry Point and Algorithm Dispatch

`main.cu` selects a CUDA device (compute capability 6.0+), parses CLI args via `ArgumentParser`, and dispatches to the appropriate algorithm function based on `--type` and `--model`:
- Model 0 (Ascetic): `bfs_opt()`, `cc_opt()`, `sssp_opt()`, `pr_opt()` in `CalculateOpt.cu`
- Model 7 (Liberator): `newbfs_opt()`, `newsssp_opt()`, `newpr_opt()` in `NewCalculateOpt.cu`; `New_CC_opt()` in `New_CC_opt.cuh`

### Core Data Structure: `GraphMeta<T>`

`GraphMeta.cuh/cu` is the central class. It manages:
- Graph storage (node pointers, edge arrays, degree arrays) on both host and device
- Static/overload partition bookkeeping
- CUDA memory allocation and host↔device transfers
- Active node tracking with Thrust-based reductions and prefix scans

Key lifecycle: `readDataFromFile()` → `initGraphHost()` → `initGraphDevice()` → `setPrestoreRatio(adviseK, nodeParamSize)` → main iteration loop.

### GPU Kernels

`gpu_kernels.cu/cuh` contains ~30+ CUDA kernels using grid-stride loops. Default execution config: 56 blocks × 1024 threads. Kernels handle node labeling, edge processing for each algorithm, and partition management.

### Algorithm Implementations

Each algorithm file (`bfs.cu`, `sssp.cu`, `cc.cu`, ~1800–2300 LOC each) implements the iteration loop:
1. Label active nodes in static partition (GPU kernel)
2. Process static partition edges (GPU kernel)
3. Count active overload nodes (Thrust reduce)
4. Stream overload edges from CPU → GPU
5. Process overload partition edges (GPU kernel)
6. Repeat until convergence

### Supporting Files

- `globals.cuh` — Type definitions (`SIZE_TYPE`, `EDGE_POINTER_TYPE`), enums (`ALG_TYPE`), structs (`EdgeWithWeight`, `FragmentData`)
- `constants.cuh` — Hardcoded test data file paths
- `TimeRecord.cu/cuh` — Performance timing (ms/µs/ns precision)
- `ArgumentParser.cu/cuh` — CLI argument parsing
- `common.cu/cuh` — Shared utilities
- `range.cuh` — Iterator-like range utilities
- `converter/converter.cpp` — Text→binary graph format converter (supports bcsr, bcsc, bwcsr, and large-scale 64-bit variants)

### Dependencies

Linked libraries: CUBLAS (`-lcublas`), CURAND (`-lcurand`). Uses Thrust (bundled with CUDA toolkit) extensively for reductions and scans.
