#include <queue>

#include "bfs.h"

void bfs_seq(int start, Graph& graph, std::vector<int>& distance)
{
	fill(distance.begin(), distance.end(), INT_MAX);
	distance[start] = 0;

	std::queue<int> to_visit;
	to_visit.push(start);

	while (!to_visit.empty())
	{
		int actual = to_visit.front();
		to_visit.pop();

		for (int i = graph.edges_offset[actual]; i < graph.edges_offset[actual] + graph.edges_size[actual]; i++)
		{
			int v = graph.adjacency_list[i];

			if (distance[v] == INT_MAX)
			{
				distance[v] = distance[actual] + 1;
				to_visit.push(v);
			}
		}
	}
}
