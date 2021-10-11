#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Timer.cuh"
#include "CheckError.cuh"

#include "bfs_noshared.cuh"
#include "graph.h"

using namespace timer;

__global__
void bfs_gpu_noshared_kernel(int* adjacency_list, int* edges_offset, int* edge_size, int* distance, int current_list_size, int* current_list, int* next_list_size, int* next_list, int level) {
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < current_list_size)
	{
		int current = current_list[idx];

		int max = edges_offset[current] + edge_size[current];
		for (int i = edges_offset[current]; i < max; i++)
		{
			int v = adjacency_list[i];

			if (distance[v] == INT_MAX)
			{
				distance[v] = level + 1;
				int position = atomicAdd(next_list_size, 1);

				next_list[position] = v;
			}
		}
	}
}

void bfs_gpu_noshared(int start, Graph& G, std::vector<int>& distance, int THREAD_BLOCK_SIZE) {
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
		bfs_gpu_noshared_kernel << <n_blocks, THREAD_BLOCK_SIZE >> > (d_adjacency_list, d_edges_offset, d_edges_size, d_distance, current_list_size, d_currentQueue, d_next_list_size, d_nextQueue, level);
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
