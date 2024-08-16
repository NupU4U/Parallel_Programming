#ifndef KRUSKAL_HPP
#define KRUSKAL_HPP

#include <mpi.h>

// Structure representing a disjoint-set (or union-find) data structure
typedef struct Set {
    int elements;
    int* canonicalElements;
    int* rank;
} Set;

// Structure representing a weighted graph
typedef struct WeightedGraph1 {
    int edges;
    int vertices;
    int* edgeList;  // Each edge is represented by 3 integers: [from, to, weight]
} WeightedGraph1;

// Function to initialize a new weighted graph
void newWeightedGraph(WeightedGraph1* graph, const int vertices, const int edges);

// Function to read graph data and generate edge list
void readGraphFile(WeightedGraph1* graph, int n);

// Function to print a weighted graph
void printWeightedGraph(const WeightedGraph1* graph);

// Function to initialize a new disjoint-set
void newSet(Set* set, const int elements);

// Function to find the root (or canonical element) of an element in the disjoint-set
int findSet(const Set* set, const int vertex);

// Function to perform union operation on two sets
void unionSet(Set* set, const int parent1, const int parent2);

// Function to delete a disjoint-set
void deleteSet(Set* set);

// Function to delete a weighted graph
void deleteWeightedGraph(WeightedGraph1* graph);

// Function to perform merge sort on the edge list of a weighted graph
void mergeSort(int* edgeList, const int start, const int end);

// Function to perform parallel sorting of edge list using MPI
void sort(WeightedGraph1* graph);

// Function to compute Minimum Spanning Tree (MST) using Kruskal's algorithm
void mstKruskal(WeightedGraph1* graph, WeightedGraph1* mst);

#endif // KRUSKAL_HPP

