# MPI Maze Generator and Solver

## Project Directory structure

project-root/
│
├── src/
│   ├── generate/
│   │   ├── bfs.cpp
│   │   ├── kruskal.cpp
│   │   └── mazegenerator.cpp
│   │
│   └── solver/
│       ├── dfs.cpp
│       ├── dijkstra.cpp
│       └── mazesolver.cpp
│
├── maze.cpp
├── Makefile
└── README.md

## Commands to Run

* Run 'make' to compile all files
* Enter the command to run the algorithms, example: mpirun -np 4 ./maze.out -g [bfs/kruskal] -s [dfs/dijkstra]
* 'make clean' to clear the build
