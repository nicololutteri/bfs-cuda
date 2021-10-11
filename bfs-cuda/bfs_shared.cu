#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Timer.cuh"
#include "CheckError.cuh"

#include "bfs_shared.cuh"
#include "graph.h"

using namespace timer;

#define MAX_EDGES 20
#define MAX_QUEUE 10000

__global__
void bfs_gpu_shared_kernel(int* adjacencyList, int* edgesOffset, int* edgesSize, int* distance, int queueSize, int* currentQueue, int* nextQueueSize, int* nextQueue, int level) {
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int blockposition = threadIdx.x;

	if (idx < queueSize)
	{
		int current = currentQueue[idx];

		__shared__ int SHARED[MAX_QUEUE];

		SHARED[blockposition * MAX_EDGES + 0] = edgesSize[current];
		for (int i = 0; i < SHARED[blockposition * MAX_EDGES + 0]; i++)
		{
			SHARED[blockposition * MAX_EDGES + i + 1] = adjacencyList[edgesOffset[current] + i];
		}

		//__syncthreads();

		for (int i = 0; i < SHARED[blockposition * MAX_EDGES + 0]; i++)
		{
			int v = SHARED[MAX_EDGES * blockposition + i + 1];

			if (v != -1)
			{
				if (distance[v] == INT_MAX)
				{
					distance[v] = level + 1;
					int position = atomicAdd(nextQueueSize, 1);
					nextQueue[position] = v;
				}
			}
		}
	}
}

void bfs_gpu_shared(int start, Graph& G, std::vector<int>& distance, int THREAD_BLOCK_SIZE)
{
	int n_blocks = (G.num_vertices + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE;

	// -------------------------------------------------------------------------
	// DEVICE MEMORY ALLOCATION
	int* d_adjacency_list;
	int* d_edges_offset;
	int* d_edges_size;
	int* d_current_list;
	int* d_next_list;
	int* d_next_list_size;
	int* d_distance;

	int current_list_size = 1;
	int NEXT_QUEUE_SIZE = 0;
	int level = 0;

	int size = G.num_vertices * sizeof(int);
	int adjacency_size = G.adjacency_list.size() * sizeof(int);

	SAFE_CALL(cudaMalloc((void**)&d_adjacency_list, adjacency_size));
	SAFE_CALL(cudaMalloc((void**)&d_edges_offset, size));
	SAFE_CALL(cudaMalloc((void**)&d_edges_size, size));
	SAFE_CALL(cudaMalloc((void**)&d_current_list, size));
	SAFE_CALL(cudaMalloc((void**)&d_next_list, size));
	SAFE_CALL(cudaMalloc((void**)&d_next_list_size, sizeof(int)));
	SAFE_CALL(cudaMalloc((void**)&d_distance, size));

	// -------------------------------------------------------------------------
	// COPY DATA FROM HOST TO DEVICE
	SAFE_CALL(cudaMemcpy(d_adjacency_list, &G.adjacency_list[0], adjacency_size, cudaMemcpyHostToDevice));
	SAFE_CALL(cudaMemcpy(d_edges_offset, &G.edges_offset[0], size, cudaMemcpyHostToDevice));
	SAFE_CALL(cudaMemcpy(d_edges_size, &G.edges_size[0], size, cudaMemcpyHostToDevice));
	SAFE_CALL(cudaMemcpy(d_next_list_size, &NEXT_QUEUE_SIZE, sizeof(int), cudaMemcpyHostToDevice));
	SAFE_CALL(cudaMemcpy(d_current_list, &start, sizeof(int), cudaMemcpyHostToDevice));

	distance = std::vector<int>(G.num_vertices, INT_MAX);
	distance[start] = 0;
	SAFE_CALL(cudaMemcpy(d_distance, distance.data(), size, cudaMemcpyHostToDevice));

	// -------------------------------------------------------------------------
	// DEVICE EXECUTION
	int* d_currentQueue = d_current_list;
	int* d_nextQueue = d_next_list;

	Timer<DEVICE> TM_onlykernel;

	TM_onlykernel.start();

	while (current_list_size > 0)
	{
		bfs_gpu_shared_kernel << <n_blocks, THREAD_BLOCK_SIZE >> > (d_adjacency_list, d_edges_offset, d_edges_size, d_distance, current_list_size, d_currentQueue, d_next_list_size, d_nextQueue, level);
		CHECK_CUDA_ERROR;

		int* tmp = d_currentQueue;
		d_currentQueue = d_nextQueue;
		d_nextQueue = tmp;

		level++;

		SAFE_CALL(cudaMemcpy(&current_list_size, d_next_list_size, sizeof(int), cudaMemcpyDeviceToHost));
		SAFE_CALL(cudaMemcpy(d_next_list_size, &NEXT_QUEUE_SIZE, sizeof(int), cudaMemcpyHostToDevice));
	}

	TM_onlykernel.stop();
	printf("\tOnly Kernel: %g ms\n", TM_onlykernel.duration());

	// -------------------------------------------------------------------------
	// COPY DATA FROM DEVICE TO HOST
	SAFE_CALL(cudaMemcpy(&distance[0], d_distance, size, cudaMemcpyDeviceToHost));

	// -------------------------------------------------------------------------
	// DEVICE MEMORY DEALLOCATION
	SAFE_CALL(cudaFree(d_adjacency_list));
	SAFE_CALL(cudaFree(d_edges_offset));
	SAFE_CALL(cudaFree(d_edges_size));
	SAFE_CALL(cudaFree(d_current_list));
	SAFE_CALL(cudaFree(d_next_list));
	SAFE_CALL(cudaFree(d_distance));
}
