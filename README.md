## Liberator: A Data Reuse Framework for Out-of-Memory Graph Computing on GPUs
Liberator is an out-of-GPU-memory graph processing framework.

#### Compilation
To compile Liberator, you need cmake (3.17+), g++ and CUDA toolkit.
You might need to change the cuda path and `CUDA_ARCHITECTURES` according to your environment in CMakeLists.txt.

```bash
mkdir -p cmake-build-debug
cd cmake-build-debug
cmake ..
make -j$(nproc)
```

The compiled executable is `cmake-build-debug/Liberator`.

#### Input graph formats
Liberator accepts binary CSR format. Different algorithms require different formats:
- `.bcsr` — for BFS and CC
- `.bcsc` — for PageRank (PR)
- `.bwcsr` — for SSSP

#### Data format converter
The converter in the `converter/` folder converts text edge lists to binary formats.

```bash
cd converter
g++ -O2 -o converter converter.cpp

# Step 1: Convert text edge list to binary
./converter txt2byte input.txt

# Step 2: Convert binary to the desired format
./converter byte2bcsr input.toB    # for BFS/CC
./converter byte2bcsc input.toB    # for PageRank
./converter byte2bwcsr input.toB   # for SSSP
```

Note: SNAP format files need comment lines (starting with `#`) removed before conversion.

#### Running
```
./Liberator \
  --input <datapath> \
  --type <bfs|cc|sssp|pr> \
  --source <node_id>       # only for bfs and sssp, default 0 \
  --model <0|7>            # 7 is Liberator, 0 is Ascetic \
  --testTime <n>           # the algorithm will execute n times \
  --adviseK <ratio>        # controls static/overload partition ratio \
  --timing <0|1>           # 1 enables per-phase timing (adds sync overhead), default 0 \
  --memory <n>             # limit GPU memory to n GB to simulate OOM scenarios, default 0 (no limit)
```

#### Publication
[IEEE Transactions on Parallel and Distributed Systems 24 April 2023 doi: DOI 10.1109/TPDS.2023.3268662] Shiyang Li, Ruiqi Tang, Jingyu Zhu, Ziyi Zhao, Xiaoli Gong, Jin Zhang, Wen-wen Wang, and Pen-Chung Yew. Liberator: A Data Reuse Framework for Out-of-Memory Graph Computing on GPUs.
