#ifndef DFS_HPP
#define DFS_HPP

#include <vector>
#include <mpi.h>

std::vector<std::vector<char>> solvedfs(const std::vector<std::vector<char>>& maze, int rank, int size);

bool dfs(std::vector<std::vector<char>>& maze, int r, int c, int rows, int cols);

#endif

