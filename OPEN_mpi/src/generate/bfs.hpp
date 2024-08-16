#ifndef BFS_HPP
#define BFS_HPP

#include <vector>
#include <functional>

// Function to convert BFS tree represented by adjacency matrix to maze
void bfs_adj_mat_to_maze(int* bfs_adj_mat, int n, int* maze2);

// Struct definition for WeightedGraph
typedef struct WeightedGraph {
    int edges;
    int vertices;
    int* edgeList;
} WeightedGraph;

// Function declarations for WeightedGraph operations
void newWeightedGraph(WeightedGraph* graph, const int vertices, const int edges);
void printWeightedGraph(const WeightedGraph* graph);
int* create_adj_mat(int n);
void distribute_graph(int* adj_mat, int n, int rank, int size, int* local_adj_mat);

// Function to perform parallel BFS on a distributed adjacency matrix
void parallel_bfs(int* local_adj_mat, int n, int rank, int size, int* final_bfs_tree_adj_mat);

#endif /* BFS_HPP */

