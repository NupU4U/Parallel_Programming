// solver/dijkstra.hpp
#ifndef DIJKSTRA_HPP
#define DIJKSTRA_HPP

#include <vector>
#include <limits>
#include <utility>
#include <mpi.h>

std::vector<std::vector<char>> solveDij(const std::vector<std::vector<char>>& maze, int rank, int size);

#endif
