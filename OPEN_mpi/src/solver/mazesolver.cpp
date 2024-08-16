// solver/mazesolver.cpp
#include "mazesolver.hpp"
#include "dfs.hpp"
#include "dijkstra.hpp"

/*std::vector<std::vector<char>> MazeSolver::solveDFS(const std::vector<std::vector<char>>& maze, int rank, int size) {
    return DFS::solve(maze, rank, size);
}

std::vector<std::vector<char>> MazeSolver::solveDijkstra(const std::vector<std::vector<char>>& maze, int rank, int size) {
    return Dijkstra::solve(maze, rank, size);
}*/

std::vector<std::vector<char>> solveDFS(std::vector<std::vector<char>>& maze, int rank, int size) {
	return solvedfs(maze, rank, size);
}
std::vector<std::vector<char>> solveDijkstra(std::vector<std::vector<char>>& maze, int rank, int size) {
	return solveDij(maze, rank, size);
}

