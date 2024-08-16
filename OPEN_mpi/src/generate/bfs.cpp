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
#include "bfs.hpp"

using namespace std;

/*typedef struct WeightedGraph {
	int edges;
	int vertices;
	int* edgeList;
} WeightedGraph;*/

void newWeightedGraph(WeightedGraph* graph, const int vertices, const int edges) {
	graph->edges = edges;
	graph->vertices = vertices;
	graph->edgeList = (int*) calloc(edges * 3, sizeof(int));
}

int* create_adj_mat(int n){
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
}

void print_adj_mat(int* adj_mat, int n){
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            std::cout<<adj_mat[i*n+j]<<" ";
        }
        std::cout<<std::endl;
    }
    std::cout<<std::endl;
}

int get_owner(int vertex, int n, int size){
    return vertex/(n/size);
}

void distribute_graph(int* adj_mat, int n, int rank, int size, int* local_adj_mat){
    int* sendbuf =  (int*)malloc(n*n*sizeof(int));
    int* sendcounts = (int*)malloc(size*sizeof(int));
    int* displs = (int*)malloc(size*sizeof(int));
    
    // distribute graphs based on vertices

    int num_vertex_per_process = n/size;

    for(int i=0; i<size; i++){
        sendcounts[i] = num_vertex_per_process*n;
        displs[i] = i*num_vertex_per_process*n;
    }

    MPI_Scatterv(adj_mat, sendcounts, displs, MPI_INT, sendbuf, num_vertex_per_process*n, MPI_INT, 0, MPI_COMM_WORLD);

    for(int i=0; i<num_vertex_per_process; i++){
        for(int j=0; j<n; j++){
            local_adj_mat[i*n+j+displs[rank]] = sendbuf[i*n+j];
        }
    }
    
}

void printWeightedGraph(const WeightedGraph* graph) {
	printf("------------------------------------------------\n");
	for (int i = 0; i < graph->edges; i++) {
		for (int j = 0; j < 3; j++) {
			printf("%d\t", graph->edgeList[i * 3 + j]);
		}
		printf("\n");
	}
	printf("------------------------------------------------\n");
}

void bfs_adj_mat_to_maze(int* bfs_adj_mat, int n, int* maze2){
    // Create the maze array
        int maze_dim = sqrt(n)*2;
        int MAZE_SIZE= maze_dim/2;
        int** maze_final; // Declaration of maze_final

	vector<int> parent_all_final(n,-1);
	
	function<void(int, int)> dfs = [&](int node, int par) {
        	parent_all_final[node] = par; // Set the parent of current node
        	for (int neighbor = 0; neighbor < n; ++neighbor) {
            		if (bfs_adj_mat[node*n+neighbor] && neighbor != par) {
                		// If there's a connection to a neighbor and it's not the parent
                		dfs(neighbor, node); // Recursively visit the neighbor
            		}
              	}
    };

    // Start DFS traversal from the root node (node 0)
    dfs(0, -1);	
	
        std::vector<int> maze(maze_dim*maze_dim,1);
//printf("THE N IS: %d\n", n);
        for(int i = 0; i < n; i++) {
            if(parent_all_final[i] != -1) {
                int u = i;
                int v = parent_all_final[i];
                int x1 = u % MAZE_SIZE;
                int y1 = u / MAZE_SIZE;
                int x2 = v % MAZE_SIZE;
                int y2 = v / MAZE_SIZE;

                int x = 2*x1;
                int y = 2*y1;
                maze[y*maze_dim + x] = 2;
                x = 2*x2;
                y = 2*y2;
                maze[y*maze_dim + x] = 2;
                x = x1 + x2;
                y = y1 + y2;
                maze[y*maze_dim + x] = 2;

            }
        }

        // Set the bottom right corner and 
        // either one above or one to the left to 0
        maze[0] = 0;
        maze[(maze_dim-1)*maze_dim + maze_dim-1] = 4;

        // Do one of these randomly 
        std::random_device rd;
        std::mt19937 g(rd());
        std::uniform_int_distribution<int> dist(0,1);
        if(dist(g) == 0) {
            maze[(maze_dim-2)*maze_dim + maze_dim-1] = 2;
        } else {
            maze[(maze_dim-1)*maze_dim + maze_dim-2] = 2;
        }
        

        // Convert maze to 2D array
        maze_final = new int*[maze_dim];
        for(int i = 0; i < maze_dim; i++) {
            maze_final[i] = new int[maze_dim];
        }
        
        // Copy 
        for(int i = 0; i < maze_dim; i++) {
            for(int j = 0; j < maze_dim; j++) {
                maze_final[i][j] = maze[i*maze_dim + j];
            }
        }
        
        // Flip about vertical axis 
        for (int i = 0; i < maze_dim; ++i) {
            for (int j = 0; j < maze_dim / 2; ++j) {
                // Swap elements from left and right sides
                std::swap(maze_final[i][j], maze_final[i][maze_dim - 1 - j]);
            }
        }

         std::cout << "Maze : " << std::endl;
         // Flip about vertical axis 
         for (int i = 0; i < maze_dim; ++i) {
             for (int j = 0; j < maze_dim ; ++j) {
                 // Swap elements from left and right sides
                 std::cout << maze_final[i][j] << " ";
             }
             std::cout  << "\n";
         }
        //print_adj_mat(maze_final, maze_dim);
}


void parallel_bfs(int* local_adj_mat, int n, int rank, int size, int* final_bfs_tree_adj_mat){

    vector<int> current_frontier;
    vector<pair<int,int>> next_frontier;
    vector<int> level(n, -1);
    int* bfs_tree_adj_mat = (int*)malloc(n*n*sizeof(int));
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            *(bfs_tree_adj_mat + i*n + j) = 0;
        }
    }

    int curr_level = 0;
    int source = 0;
    if(rank == get_owner(source, n, size)){
        current_frontier.push_back(source);
        level[source] = 0;
    }
    // bool done = true;

    while(true){
        vector<vector<pair<int,int>>> send_buffs(size);
        while(current_frontier.size() > 0){
            // printf("Rank %d: %d\n", rank, current_frontier.size());
            int node = current_frontier.back();
            current_frontier.pop_back();

            for(int i=0; i<n; i++){
                if(*(local_adj_mat + node*n + i) == 1 ){
                    // next_frontier.push_back((make_pair(node, i)));
                    int owner = get_owner(i, n, size);
                    send_buffs[owner].push_back(make_pair(node, i));
                }
            }
        }

        int* to_send = (int*)malloc(size*n*n*sizeof(int));
        int* sendcounts = (int*)malloc(size*sizeof(int));
        int* displs = (int*)malloc(size*sizeof(int));
        for(int i=0; i<size; i++){
            for(int j=0; j<n*n; j++){
                to_send[i*n*n + j] = -1;
            }
        }
        for(int i=0; i<size; i++){
            for(int j=0; j<send_buffs[i].size(); j++){
                to_send[i*n*n + j*2] = send_buffs[i][j].first;
                to_send[i*n*n + j*2 + 1] = send_buffs[i][j].second;
            }
        }

        for(int i=0; i<size; i++){
            sendcounts[i] = n*n;
        }

        displs[0] = 0;
        for(int i=1; i<size; i++){
            displs[i] = displs[i-1] + sendcounts[i-1];
        }

        int* recv_buff = (int*)malloc(size*n*n*sizeof(int));
        MPI_Alltoallv(to_send, sendcounts, displs, MPI_INT, recv_buff, sendcounts, displs, MPI_INT, MPI_COMM_WORLD);
        // process the received data into next_frontier removing duplicates
        for(int i=0; i<size; i++){
            for(int j=0; j<n*n; j+=2){
                if(recv_buff[i*n*n + j] == -1){
                    break;
                }
                next_frontier.push_back(make_pair(recv_buff[i*n*n + j], recv_buff[i*n*n + j + 1]));
                // printf("Rank %d: %d %d\n", rank, recv_buff[i*n*n + j], recv_buff[i*n*n + j + 1]);
            }
        }

        // remove entries with same i and randomply keep one
        random_shuffle(next_frontier.begin(), next_frontier.end());
        // keep the first occurence of each parent neighbour pair

        MPI_Barrier(MPI_COMM_WORLD);
    // printf("Rank %d done\n", rank);
        vector<int> found(n, 0);
        vector<pair<int,int>> new_next_frontier;
        for(int i=0; i<next_frontier.size(); i++){
            // printf("Rank %d: %d %d\n", rank, next_frontier[i].first, next_frontier[i].second);
            if(found[next_frontier[i].first] == 0){
                new_next_frontier.push_back(next_frontier[i]);
                found[next_frontier[i].first] = 1;
            }
        }

        current_frontier.clear();
        for(int i=0; i<new_next_frontier.size(); i++){
            if(level[new_next_frontier[i].second] == -1){
                level[new_next_frontier[i].second] = curr_level + 1;
                current_frontier.push_back(new_next_frontier[i].second);
                *(bfs_tree_adj_mat + new_next_frontier[i].first*n + new_next_frontier[i].second) = 1;
                *(bfs_tree_adj_mat + new_next_frontier[i].second*n + new_next_frontier[i].first) = 1;
            }
        }

        next_frontier.clear();
        curr_level++;
        int size_frontier = current_frontier.size();
        int haha = size_frontier;
        MPI_Allreduce(&size_frontier, &haha, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        // printf("here\n");
        // printf("Rankgf fb %d: %d\n", rank, haha);
        if(haha == 0){
            break;
        }

        free(to_send);
        free(recv_buff);
        free(sendcounts);
        free(displs);
    }
    
    // merge the bfs_tree_adj_mat from all processes
    MPI_Reduce(bfs_tree_adj_mat, final_bfs_tree_adj_mat, n*n, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if(rank == 0){
        // MPI_Barrier(MPI_COMM_WORLD);
        //printf("ADJ MATTT\n");
        //print_adj_mat(final_bfs_tree_adj_mat, n);
        int* maze = (int*)malloc(2*n*2*n*sizeof(int));
        bfs_adj_mat_to_maze(final_bfs_tree_adj_mat, n, maze);
    }
        
}

/*int main(int argc, char** argv){
    int rank, size, n;
    int* adj_mat;
	srand(time(NULL));
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    WeightedGraph tempMST = { .edges = 0, .vertices = 0, .edgeList = NULL };
    WeightedGraph* mst = &tempMST;
    vector<vector<int>> edgess ;


    if(rank == 0){
        n = int(atoi(argv[1]));
        adj_mat = create_adj_mat(n);
        // print_adj_mat(adj_mat, n);
    }
    int* final_bfs_tree_adj_mat = (int*)malloc(n*n*sizeof(int));
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int* local_adj_mat = (int*)calloc(n*n, sizeof(int));
    distribute_graph(adj_mat, n, rank, size, local_adj_mat);
    // print_adj_mat(local_adj_mat, n);
    parallel_bfs(local_adj_mat, n, rank, size, final_bfs_tree_adj_mat);

    if (rank == 0) {
        int vertices = n*n;
        int edges = 2*n*(n-1);
        newWeightedGraph(mst, vertices, edges);
        //storing  the final bfs tree in  edges of mst such that mst->edgeList[i] = u, mst->edgeList[i+1] = v and their is no weight in bfs tree
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (*(final_bfs_tree_adj_mat + i*n + j) == 1) {
                    vector<int> edge;
                    edge.push_back(i);
                    edge.push_back(j);
                    edgess.push_back(edge);
                }
            }
        }
    }
    if (rank == 0) {

		int nm = sqrt(n)*2;

		char maze[nm][nm];
		for(int i=0; i<nm; i++){
			for(int j=0; j<nm; j++){
				maze[i][j] = '*';
			}
		}

		for(int i=0; i<nm; i+=2){
			for(int j=0; j<nm; j+=2){
				maze[i][j] = '0';
			}
		}

		for(int i=0; i<nm; i+=2){
			maze[i][nm-1] = '0';
			maze[nm-1][i] = '0';
		}
		unsigned long weightMST = 0;
		for (int i = 0; i < edgess.size(); i++) {
			// weightMST += mst->edgeList[i * 3 + 2];
			int n1 = edgess[i][0];
			int n2 = edgess[i][1];
			int n = nm/2;
            //sqrt n
			int i1 = n1/n;
			int j1 = n1%n;
			int i2 = n2/n;
			int j2 = n2%n;
			maze[i1+i2][j1+j2] = '0';
		}
		// printf("MST weight: %lu\n", weightMST);
        //print edges 
        //print adj matrix
        // print_adj_mat(final_bfs_tree_adj_mat, n);
        // for (int i = 0; i < edgess.size(); i++) {
        //     printf("%d %d\n", edgess[i][0], edgess[i][1]);
        // }
		// for(int i=0; i<nm; i++){
		// 	for(int j=0; j<nm; j++){
		// 		printf("%c", maze[i][j]);
		// 	}
		// 	printf("\n");
		// }
    }

    MPI_Finalize();
    return 0;
}
*/




