#ifndef BFS_CPU_H
#define BFS_CPU_H

#include <vector>

#include "graph.h"

void bfs_seq(int start, Graph& G, std::vector<int>& distance);

#endif // BFS_CPU_H
