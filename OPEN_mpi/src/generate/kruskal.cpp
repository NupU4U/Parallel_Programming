#include <iostream>
#include <fstream>
#include <mpi.h>
#include <stdlib.h>
#include <cstring>
#include <vector>
#include <queue>
#include <vector>
#include <utility>
#include <time.h>
#include <set>
#include <algorithm>
#include <random>
#include <stdio.h>
#include "kruskal.hpp"

using namespace std;

const int UNSET_ELEMENT = -1;

/*typedef struct Set {
	int elements;
	int* canonicalElements;
	int* rank;
} Set;

typedef struct WeightedGraph {
	int edges;
	int vertices;
	int* edgeList;
} WeightedGraph;
*/

void newWeightedGraph(WeightedGraph1* graph, const int vertices, const int edges) {
	graph->edges = edges;
	graph->vertices = vertices;
	graph->edgeList = (int*) calloc(edges * 3, sizeof(int));
}

/*int* create_adj_mat(int n){
    int* adj_mat = (int*)malloc(n*n*sizeof(int));
    for(int i=0; i<n; i++){
        for(int j=i; j<n; j++){
            int temp = rand()%10;
            if(temp == 0){
                adj_mat[i*n+j] = 1;
                adj_mat[j*n+i] = 1;
            }else{
                adj_mat[i*n+j] = 1;
                adj_mat[j*n+i] = 1;
            }
        }
    }
    return adj_mat;
}*/

void readGraphFile(WeightedGraph1* graph, int n) {
    int vertices = n;
    int edges = n * (n - 1) / 2; // For a fully connected graph, every pair of vertices is connected, so total edges are n*(n-1)/2
    newWeightedGraph(graph, vertices, edges);
    
    int edgeIndex = 0; // Keep track of the current edge index
    
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) { // Loop from i+1 to n to avoid duplicate edges and self-loops
            // Connect vertex i and vertex j with a random weight
            graph->edgeList[edgeIndex * 3] = i;
            graph->edgeList[edgeIndex * 3 + 1] = j;
            graph->edgeList[edgeIndex * 3 + 2] = 1 + (rand() % 10000);
            edgeIndex++;
        }
    }
    
}

void printWeightedGraph(const WeightedGraph1* graph) {
	printf("------------------------------------------------\n");
	for (int i = 0; i < graph->edges; i++) {
		for (int j = 0; j < 3; j++) {
			printf("%d\t", graph->edgeList[i * 3 + j]);
		}
		printf("\n");
	}
	printf("------------------------------------------------\n");
}

void newSet(Set* set, const int elements) {
	set->elements = elements;
	set->canonicalElements = (int*) malloc(elements * sizeof(int));
	memset(set->canonicalElements, UNSET_ELEMENT, elements * sizeof(int));
	set->rank = (int*) calloc(elements, sizeof(int));
}

int findSet(const Set* set, const int vertex) {
	if (set->canonicalElements[vertex] == UNSET_ELEMENT) {
		return vertex;
	} 
	else {
		set->canonicalElements[vertex] = findSet(set,set->canonicalElements[vertex]);
		return set->canonicalElements[vertex];
	}
}

void unionSet(Set* set, const int parent1, const int parent2) {
	int root1 = findSet(set, parent1);
	int root2 = findSet(set, parent2);

	if (root1 == root2) {
		return;
	} 
	else if (set->rank[root1] < set->rank[root2]) {
		set->canonicalElements[root1] = root2;
	} 
	else if (set->rank[root1] > set->rank[root2]) {
		set->canonicalElements[root2] = root1;
	} 
	else {
		set->canonicalElements[root1] = root2;
		set->rank[root2] = set->rank[root1] + 1;
	}
}

void copyEdge(int* to, int* from) {
	memcpy(to, from, 3 * sizeof(int));
}


void scatterEdgeList(int* edgeList, int* edgeListPart, const int elements,int* elementsPart) {
	int rank;
	int size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	MPI_Scatter(edgeList, *elementsPart * 3, MPI_INT, edgeListPart,	*elementsPart * 3, MPI_INT, 0, MPI_COMM_WORLD);

	if (rank == size - 1 && elements % *elementsPart != 0) {
		*elementsPart = elements % *elementsPart;
	}

	if (elements / 2 + 1 < size && elements != size) {
		if (rank == 0) {
			fprintf(stderr, "Unsupported size/process combination, exiting!\n");
		}
		MPI_Finalize();
		exit(EXIT_FAILURE);
	}
}

void deleteSet(Set* set) {
	free(set->canonicalElements);
	free(set->rank);
}

void deleteWeightedGraph(WeightedGraph1* graph) {
	free(graph->edgeList);
}

void merge(int* edgeList, const int start, const int end, const int pivot) {
	int length = end - start + 1;
	int* working = (int*) malloc(length * 3 * sizeof(int));

	memcpy(working, &edgeList[start * 3],(pivot - start + 1) * 3 * sizeof(int));

	int workingEnd = end + pivot - start + 1;
	for (int i = pivot + 1; i <= end; i++) {
		copyEdge(&working[(workingEnd - i) * 3],&edgeList[i * 3]);
	}

	int left = 0;
	int right = end - start;
	for (int k = start; k <= end; k++) {
		if (working[right * 3 + 2]< working[left * 3 + 2]) {
			copyEdge(&edgeList[k * 3],&working[right * 3]);
			right--;
		} else {
			copyEdge(&edgeList[k * 3],&working[left * 3]);
			left++;
		}
	}

	free(working);
}

void mergeSort(int* edgeList, const int start, const int end) {
	if (start != end) {
		int pivot = (start + end) / 2;
		mergeSort(edgeList, start, pivot);
		mergeSort(edgeList, pivot + 1, end);

		merge(edgeList, start, end, pivot);
	}
}

void sort(WeightedGraph1* graph) {
	int rank;
	int size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Status status;

	bool parallel = size != 1;

	int elements;
	if (rank == 0) {
		elements = graph->edges;
		MPI_Bcast(&elements, 1, MPI_INT, 0, MPI_COMM_WORLD);
	} 
	else {
		MPI_Bcast(&elements, 1, MPI_INT, 0, MPI_COMM_WORLD);
	}

	int elementsPart = (elements + size - 1) / size;
	int* edgeListPart = (int*) malloc(elementsPart * 3 * sizeof(int));
	if (parallel) {
		scatterEdgeList(graph->edgeList, edgeListPart, elements, &elementsPart);
	} else {
		edgeListPart = graph->edgeList;
	}

	mergeSort(edgeListPart, 0, elementsPart - 1);

	if (parallel) {
		int from;
		int to;
		int elementsRecieved;
		for (int step = 1; step < size; step *= 2) {
			if (rank % (2 * step) == 0) {
				from = rank + step;
				if (from < size) {
					MPI_Recv(&elementsRecieved, 1, MPI_INT, from, 0,MPI_COMM_WORLD, &status);
					edgeListPart = (int*)realloc(edgeListPart,(elementsPart + elementsRecieved) * 3* sizeof(int));
					MPI_Recv(&edgeListPart[elementsPart * 3],elementsRecieved * 3,MPI_INT, from, 0, MPI_COMM_WORLD, &status);
					merge(edgeListPart, 0, elementsPart + elementsRecieved - 1,	elementsPart - 1);
					elementsPart += elementsRecieved;
				}
			} 
			else if (rank % step == 0) {
				to = rank - step;
				MPI_Send(&elementsPart, 1, MPI_INT, to, 0, MPI_COMM_WORLD);
				MPI_Send(edgeListPart, elementsPart * 3, MPI_INT, to,0,	MPI_COMM_WORLD);
			}
		}

		if (rank == 0) {
			free(graph->edgeList);
			graph->edgeList = edgeListPart;
		} else {
			free(edgeListPart);
		}
	} else {
		graph->edgeList = edgeListPart;
	}
}

void mstKruskal(WeightedGraph1* graph, WeightedGraph1* mst) {
	Set tempSet = { .elements = 0, .canonicalElements = NULL, .rank = NULL };
    Set* set = &tempSet;
	newSet(set, graph->vertices);

	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	sort(graph);

	if (rank == 0) {
		int currentEdge = 0;
		for (int edgesMST = 0;edgesMST < graph->vertices - 1 || currentEdge < graph->edges;) {
			int canonicalElementFrom = findSet(set,	graph->edgeList[currentEdge * 3]);
			int canonicalElementTo = findSet(set,graph->edgeList[currentEdge * 3 + 1]);
			if (canonicalElementFrom != canonicalElementTo) {
				copyEdge(&mst->edgeList[edgesMST * 3],&graph->edgeList[currentEdge * 3]);
				unionSet(set, canonicalElementFrom, canonicalElementTo);
				edgesMST++;
			}
			currentEdge++;
		}
	}

	deleteSet(set);
}
/*
int main(int argc, char* argv[]) {
    int rank, size, n;
    // int* adj_mat;
	srand(time(0));
	// rand(time(0));
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	
	WeightedGraph tempGraph = { .edges = 0, .vertices = 0, .edgeList = NULL };
    WeightedGraph tempMST = { .edges = 0, .vertices = 0, .edgeList = NULL };
    WeightedGraph* graph = &tempGraph;
    WeightedGraph* mst = &tempMST;
    vector<vector<int>> edgess ;

	if (rank == 0) {
		n = int(atoi(argv[1]));
		readGraphFile(graph, n);
		newWeightedGraph(mst, graph->vertices, graph->vertices - 1);
	}
	double start = MPI_Wtime();
	mstKruskal(graph, mst);
	int adj_matri[n][n];
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			adj_matri[i][j] = 0;
		}
	}
	if (rank == 0) {

		int nm = n*2;

		char maze[nm][nm];
		for(int i=0; i<nm; i++){
			for(int j=0; j<nm; j++){
				maze[i][j] = '0';
			}
		}

		for(int i=0; i<nm; i+=2){
			for(int j=0; j<nm; j+=2){
				maze[i][j] = '1';
			}
		}

		for(int i=0; i<nm; i+=2){
			maze[i][nm-1] = '1';
			maze[nm-1][i] = '1';
		}
		unsigned long weightMST = 0;
		for (int i = 0; i < mst->edges; i++){
			weightMST += mst->edgeList[i * 3 + 2];
			int n1 = mst->edgeList[i*3];
			int n2 = mst->edgeList[i*3 + 1];
			int n = nm/2;
			int i1 = n1/n;
			int j1 = n1%n;
			int i2 = n2/n;
			int j2 = n2%n;
			maze[i1+i2][j1+j2] = '1';
		}
		printf("MST weight: %lu\n", weightMST);
		for (int i = 0; i < mst->edges; i++) {
			adj_matri[mst->edgeList[i*3]][mst->edgeList[i*3 + 1]] = 1;
			adj_matri[mst->edgeList[i*3 + 1]][mst->edgeList[i*3]] = 1;
		}
		//print adj matrix
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				printf("%d ", adj_matri[i][j]);
			}
			printf("\n");
		}
		
		deleteWeightedGraph(graph);
		deleteWeightedGraph(mst);
		printf("Time elapsed: %f s\n", MPI_Wtime() - start);
	}
	MPI_Finalize();

	return EXIT_SUCCESS;
}*/
