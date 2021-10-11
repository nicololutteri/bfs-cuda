#ifndef BFS_SHARED_CUH
#define BFS_SHARED_CUH

#include "graph.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void bfs_gpu_shared_kernel(int*, int*, int*, int*, int, int*, int*, int*, int);
void bfs_gpu_shared(int, Graph&, std::vector<int>&, int N_THREADS_PER_BLOCK);

#endif // BFS_SHARED_CUH
