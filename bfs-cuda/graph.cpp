#include <stdio.h>
#include <vector>

#include "graph.h"

Graph::Graph() {
	int num_vertices;
	int num_edges;

	int result = scanf("%i %i", &num_vertices, &num_edges);

	std::vector<std::vector<int>> adjacency_list(num_vertices);

	for (int i = 0; i < num_edges; i++)
	{
		int first;
		int second;

		result = scanf("%i %i", &first, &second);

		if (first < num_vertices && second < num_vertices)
		{
			adjacency_list[first].push_back(second);
		}
		else
		{
			printf("Error edges not valid: %d %d\n", first, second);
		}
	}

	create_arrays(adjacency_list, num_edges);
}

void Graph::create_arrays(std::vector<std::vector<int>> adjacency_list, int num_edges) {
	for (int i = 0; i < adjacency_list.size(); i++)
	{
		this->edges_offset.push_back(this->adjacency_list.size());
		this->edges_size.push_back(adjacency_list[i].size());

		for (int v : adjacency_list[i])
		{
			this->adjacency_list.push_back(v);
		}
	}

	this->num_vertices = adjacency_list.size();
	this->num_edges = num_edges;
}
