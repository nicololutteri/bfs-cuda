#ifndef BFS_NOSHARED_CUH
#define BFS_NOSHARED_CUH

#include "graph.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void bfs_gpu_noshared_kernel(int*, int*, int*, int*, int, int*, int*, int*, int);
void bfs_gpu_noshared(int start, Graph&, std::vector<int>&, int N_THREADS_PER_BLOCK);

#endif // BFS_NOSHARED_CUH
