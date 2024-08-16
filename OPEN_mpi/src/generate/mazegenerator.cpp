#include "mazegenerator.hpp"
#include <vector>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <mpi.h>
#include "bfs.hpp"
#include "kruskal.hpp"

using namespace std;

void generateMazeBFS(int n) {
    int* adj_mat;
    int rank, size;
    srand(time(NULL));
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    WeightedGraph tempMST = { .edges = 0, .vertices = 0, .edgeList = NULL };
    WeightedGraph* mst = &tempMST;
    vector<vector<int>> edgess;
    
    if (rank == 0) {
        adj_mat = create_adj_mat(n);
    }

    int* final_bfs_tree_adj_mat = (int*)malloc(n * n * sizeof(int));
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int* local_adj_mat = (int*)calloc(n * n, sizeof(int));
    distribute_graph(adj_mat, n, rank, size, local_adj_mat);

    parallel_bfs(local_adj_mat, n, rank, size, final_bfs_tree_adj_mat);

    if (rank == 0) {
        int vertices = n * n;
        int edges = 2 * n * (n - 1);
        newWeightedGraph(mst, vertices, edges);

        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (*(final_bfs_tree_adj_mat + i * n + j) == 1) {
                    vector<int> edge;
                    edge.push_back(i);
                    edge.push_back(j);
                    edgess.push_back(edge);
                }
            }
        }

        int maze_dim = sqrt(n) * 2;
        vector<vector<char>> maze(maze_dim, vector<char>(maze_dim, '*'));

        for (int i = 0; i < maze_dim; i += 2) {
            for (int j = 0; j < maze_dim; j += 2) {
                maze[i][j] = '0';
            }
        }

        for (int i = 0; i < maze_dim; i += 2) {
            maze[i][maze_dim - 1] = '0';
            maze[maze_dim - 1][i] = '0';
        }

        for (int i = 0; i < edgess.size(); i++) {
            int n1 = edgess[i][0];
            int n2 = edgess[i][1];
            int block_size = maze_dim / 2;
            int i1 = n1 / block_size;
            int j1 = n1 % block_size;
            int i2 = n2 / block_size;
            int j2 = n2 % block_size;
            maze[i1 + i2][j1 + j2] = '0';
        }

        cout << "Maze:" << endl;
        for (int i = 0; i < maze_dim; i++) {
            for (int j = 0; j < maze_dim; j++) {
                cout << maze[i][j] << " ";
            }
            cout << endl;
        }
    }

  //  MPI_Finalize();
}

void generateMazeKruskal(int n) {
    int rank, size;
    // int* adj_mat;
	srand(time(0));
	// rand(time(0));
	MPI_Init(NULL, NULL);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	
	WeightedGraph1 tempGraph = { .edges = 0, .vertices = 0, .edgeList = NULL };
    	WeightedGraph1 tempMST = { .edges = 0, .vertices = 0, .edgeList = NULL };
    	WeightedGraph1* graph = &tempGraph;
    	WeightedGraph1* mst = &tempMST;
    	vector<vector<int>> edgess ;

	if (rank == 0) {
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
//	MPI_Finalize();

//	return EXIT_SUCCESS;
}

