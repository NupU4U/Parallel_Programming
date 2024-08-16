// solver/dijkstra.cpp
#include "dijkstra.hpp"
#include <vector>
#include <limits>
#include <utility>
#include <queue>

std::vector<std::vector<char>> solveDij(const std::vector<std::vector<char>>& maze, int rank, int size) {
    int rows = maze.size();
    int cols = maze[0].size();
    std::vector<std::vector<char>> final_maze = maze;

    struct Node {
        int row;
        int col;
        int distance;
        bool operator>(const Node& other) const {
            return distance > other.distance;
        }
    };

    std::vector<std::vector<int>> distance(rows, std::vector<int>(cols, std::numeric_limits<int>::max()));
    std::vector<std::vector<bool>> visited(rows, std::vector<bool>(cols, false));
    std::vector<std::vector<std::pair<int, int>>> prev(rows, std::vector<std::pair<int, int>>(cols));

    std::priority_queue<Node, std::vector<Node>, std::greater<Node>> pq;
    pq.push({0, cols - 1, 0});
    distance[0][cols - 1] = 0; 
    
    // Dijkstra's algorithm
    while (!pq.empty()) {
        Node current = pq.top();
        pq.pop();
        int r = current.row;
        int c = current.col;
        int dist = current.distance;

        if (visited[r][c]) {
            continue;
        }

        visited[r][c] = true;

        // Explore neighbors
        if (r == rows - 1 && c == 0) {
            while (!(r == 0 && c == cols - 1)) {
                int prev_r = prev[r][c].first;
                int prev_c = prev[r][c].second;
                final_maze[r][c] = 'P';
                r = prev_r;
                c = prev_c;
            }
            final_maze[0][cols - 1] = 'S'; // Mark start cell
            final_maze[rows - 1][0] = 'E'; // Mark exit cell
            break;
        }

        if (r - 1 >= 0 && maze[r - 1][c] == ' ' && dist + 1 < distance[r - 1][c]) {
            distance[r - 1][c] = dist + 1;
            prev[r - 1][c] = {r, c};
            pq.push({r - 1, c, dist + 1});
        }
        if (r + 1 < rows && maze[r + 1][c] == ' ' && dist + 1 < distance[r + 1][c]) {
            distance[r + 1][c] = dist + 1;
            prev[r + 1][c] = {r, c};
            pq.push({r + 1, c, dist + 1});
        }
        if (c - 1 >= 0 && maze[r][c - 1] == ' ' && dist + 1 < distance[r][c - 1]) {
            distance[r][c - 1] = dist + 1;
            prev[r][c - 1] = {r, c};
            pq.push({r, c - 1, dist + 1});
        }
        if (c + 1 < cols && maze[r][c + 1] == ' ' && dist + 1 < distance[r][c + 1]) {
            distance[r][c + 1] = dist + 1;
            prev[r][c + 1] = {r, c};
            pq.push({r, c + 1, dist + 1});
        }
    }

  /*  for (int i = 0; i < size; ++i) {
        if (i != rank) {
            MPI_Send(&final_maze[0][0], rows * cols, MPI_CHAR, i, 0, MPI_COMM_WORLD);
        }
    }

    // Receive solved maze from rank 0
    if (rank != 0) {
        MPI_Status status;
        MPI_Recv(&final_maze[0][0], rows * cols, MPI_CHAR, 0, 0, MPI_COMM_WORLD, &status);
    }*/

    return final_maze;
}
