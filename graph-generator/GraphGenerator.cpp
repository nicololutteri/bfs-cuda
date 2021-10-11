#include <iostream>
#include <string>
#include <vector>

#include "GraphGenerator.h"

int main(int argc, char* argv[])
{
	int nodes = std::stoi(argv[1]);
	int edgespernode = std::stoi(argv[2]);

	printf("%d %d\n", nodes * edgespernode, nodes * edgespernode);

	int from = 0;
	for (int i = 0; i < nodes; i++)
	{
		int to = from * edgespernode + 1;

		for (int j = 0; j < edgespernode && to < nodes * edgespernode; j++)
		{
			printf("%d %d\n", from, to);
			to++;
		}

		from++;
	}
}
