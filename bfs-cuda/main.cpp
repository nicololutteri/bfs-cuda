#include "Timer.cuh"
#include "CheckError.cuh"

#include "graph.h"

#include "bfs.h"
#include "bfs_shared.cuh"
#include "bfs_noshared.cuh"

#include "main.h"

using namespace timer;

int main(int argc, char* argv[]) 
{
	int THREAD_BLOCK_SIZE = std::stoi(argv[1]);
	if (argc == 3)
	{
		freopen(argv[2], "r", stdin);
	}

	// --------------------------------------------------------------------------
	// HOST INITILIZATION
	std::cout << "Reading..." << std::endl;
	Graph G{};
	std::cout << "End graph" << std::endl << std::endl;

	int start_vertex = 0;

	// --------------------------------------------------------------------------
	// CPU

	Timer<HOST>   TM_host;

	std::vector<int> distance_seq = std::vector<int>(G.num_vertices);

	// -------------------------------------------------------------------------
	// HOST EXECUTION
	TM_host.start();
	bfs_seq(start_vertex, G, distance_seq);
	TM_host.stop();

	printf("CPU Seq: %g ms\n\n", TM_host.duration());

	// -------------------------------------------------------------------------
	// NO SHARED GPU

	Timer<DEVICE> TM_device_no_shared;

	std::vector<int> distance = std::vector<int>(G.num_vertices);

	// -------------------------------------------------------------------------
	// DEVICE EXECUTION
	TM_device_no_shared.start();
	bfs_gpu_noshared(start_vertex, G, distance, THREAD_BLOCK_SIZE);
	TM_device_no_shared.stop();

	// -------------------------------------------------------------------------
	// RESULT CHECK
	checkArray(distance_seq, distance);

	printf("GPU No Shared: %g ms\n", TM_device_no_shared.duration());
	std::cout << std::setprecision(1) << "Speedup: " << TM_host.duration() / TM_device_no_shared.duration() << "x\n\n";

	// -------------------------------------------------------------------------
	// SHARED GPU

	Timer<DEVICE> TM_device_shared;

	distance = std::vector<int>(G.num_vertices);

	// -------------------------------------------------------------------------
	// DEVICE EXECUTION
	TM_device_shared.start();
	bfs_gpu_shared(start_vertex, G, distance, THREAD_BLOCK_SIZE);
	TM_device_shared.stop();

	// -------------------------------------------------------------------------
	// RESULT CHECK
	checkArray(distance_seq, distance);

	printf("GPU Shared: %g ms\n", TM_device_shared.duration());
	std::cout << std::setprecision(1) << "Speedup: " << TM_host.duration() / TM_device_shared.duration() << "x\n\n";

	return 0;
}

bool checkArray(std::vector<int> original, std::vector<int> to_check)
{
	if (original.size() != to_check.size())
	{
		return false;
	}

	for (int i = 0; i < original.size(); i++)
	{
		if (original[i] != to_check[i])
		{
			printf("Error at %d (%d != %d)\n", i, original[i], to_check[i]);
			return false;
		}
	}

	return true;
}
