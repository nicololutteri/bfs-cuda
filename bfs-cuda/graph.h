#ifndef GRAPH_H
#define GRAPH_H

#include <vector>

class Graph
{
public:
	Graph();

	int num_vertices;
	int num_edges;

	std::vector<int> adjacency_list;

	std::vector<int> edges_offset;
	std::vector<int> edges_size;

private:
	void create_arrays(std::vector<std::vector<int>> adjacency_list, int num_edges);
};

#endif // GRAPH_H
