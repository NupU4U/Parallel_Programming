#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cstring>
#include <vector>
#include <mpi.h>
#include <fstream>
#include "generate/mazegenerator.hpp"
#include "solver/mazesolver.hpp"

using namespace std;

int main(int argc, char** argv) {
    int rank, size, n;
    n = 64;
    
    const char* option = nullptr; const char* option2=nullptr;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-g") == 0 && i + 1 < argc) {
            option = argv[i + 1];
        }
        if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            option2 = argv[i + 1];
        }
        
    }

    if (option == nullptr) {
        std::cerr << "Error: -g option not provided or invalid." << std::endl;
        return EXIT_FAILURE;
    }

    // Now 'option' contains the value of the -g argument (either "bfs" or "kruskal")
    if (strcmp(option, "bfs") == 0) {
        // Execute BFS algorithm
        std::cout << "Executing BFS algorithm..." << std::endl;
        generateMazeBFS(n);
    } 
    else if (strcmp(option, "kruskal") == 0) {
        // Execute Kruskal's algorithm
        std::cout << "Executing Kruskal's algorithm..." << std::endl;
        generateMazeKruskal(n);
        // Add your Kruskal's algorithm implementation here
    }
    if (option2 == nullptr) {
        std::cerr << "Error: -s option not provided or invalid." << std::endl;
        return EXIT_FAILURE;
    }

    if (strcmp(option2, "dfs") == 0) {
        // Execute BFS algorithm
        std::cout << "Executing DFS algorithm..." << std::endl;
        //MPI_Init(&argc, &argv);
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    	MPI_Comm_size(MPI_COMM_WORLD, &size);
    	std::vector<std::vector<char>> maze;
	if(rank==0) {
    		// Open the input file
    		std::ifstream inputFile("txtFile2.txt");
    		if (!inputFile) {
        		std::cerr << "Error: Unable to open the input file." << std::endl;
        		//MPI_Abort(MPI_COMM_WORLD, 1); // Abort MPI execution
    	}

    	// Read each line from the file and construct the maze
    	std::string line;
    	while (std::getline(inputFile, line)) {
        	if (!line.empty()) {
            	std::vector<char> row;
            	for (char c : line) {
            	        row.push_back(c);
           	 }
           	 maze.push_back(row);
      	  }
 	   }
	
   	 // Close the input file
  	  inputFile.close();
   	  std::vector<std::vector<char>> solvedMaze = solveDFS(maze,rank, size);
   	  std::cout << "Solved Maze:" << std::endl;
       	  for (const auto& row : solvedMaze) {
         	   for (char cell : row) {
           	     std::cout << cell << " ";
            }
            	std::cout << std::endl;
          }
    	}
    	MPI_Finalize();
    } 
    else if (strcmp(option2, "dijkstra") == 0) {
        // Execute Kruskal's algorithm
        std::cout << "Executing Dijkstra's algorithm..." << std::endl;        
	//MPI_Init(&argc, &argv);
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    	MPI_Comm_size(MPI_COMM_WORLD, &size);
    	std::vector<std::vector<char>> maze;
	if(rank==0) {
    		// Open the input file
    		std::ifstream inputFile("txtFile2.txt");
    		if (!inputFile) {
        		std::cerr << "Error: Unable to open the input file." << std::endl;
        		//MPI_Abort(MPI_COMM_WORLD, 1); // Abort MPI execution
    	}

    	// Read each line from the file and construct the maze
    	std::string line;
    	while (std::getline(inputFile, line)) {
        	if (!line.empty()) {
            	std::vector<char> row;
            	for (char c : line) {
            	        row.push_back(c);
           	 }
           	 maze.push_back(row);
      	  }
 	   }
	
   	 // Close the input file
  	  inputFile.close();
   	  std::vector<std::vector<char>> solvedMaze = solveDijkstra(maze,rank, size);
   	  std::cout << "Solved Maze:" << std::endl;
       	  for (const auto& row : solvedMaze) {
         	   for (char cell : row) {
           	     std::cout << cell << " ";
            }
            	std::cout << std::endl;
          }
    	}
    	MPI_Finalize();
        
    }
	
    return 0;
}

