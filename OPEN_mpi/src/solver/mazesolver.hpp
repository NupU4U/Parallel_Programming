// solver/mazesolver.hpp
#ifndef MAZESOLVER_HPP
#define MAZESOLVER_HPP

#include <vector>
#include <mpi.h>


std::vector<std::vector<char>> solveDFS( std::vector<std::vector<char>>& maze, int rank, int size);
std::vector<std::vector<char>> solveDijkstra( std::vector<std::vector<char>>& maze, int rank, int size);

#endif
