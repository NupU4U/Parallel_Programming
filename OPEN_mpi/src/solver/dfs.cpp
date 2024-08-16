#include <iostream>
#include "dfs.hpp"

// Define the solveDFS function
std::vector<std::vector<char>> solvedfs(const std::vector<std::vector<char>>& maze, int rank, int size) {
    int rows = maze.size();
    int cols = maze[0].size();
    std::vector<std::vector<char>> solvedMaze = maze;
    // Start DFS from the top right corner (0, cols-1)
    dfs(solvedMaze, 0, cols - 1, rows, cols); // Call DFS::dfs

    return solvedMaze;
}

// Define the dfs helper function
bool dfs(std::vector<std::vector<char>>& maze, int r, int c, int rows, int cols) {
    // Base case: Check if we've reached the exit (bottom left corner)
    if (r == rows - 1 && c == 0) {
        maze[r][c] = 'E'; // Mark exit cell
        maze[0][cols-1] = 'S';
        return true;
    }

    // Explore neighbors (up, down, left, right)
    if (r >= 0 && r < rows && c >= 0 && c < cols && maze[r][c] == ' ') {
        maze[r][c] = 'P'; // Mark current cell as part of the path
	
        // Recursive calls for all possible directions
        bool b1= dfs(maze, r - 1, c, rows, cols); // Up
        bool b2= dfs(maze, r + 1, c, rows, cols); // Down
        bool b3= dfs(maze, r, c - 1, rows, cols); // Left
        bool b4= dfs(maze, r, c + 1, rows, cols); // Right
        if(b1 or b2 or b3 or b4) return true;
     	maze[r][c] = ' ';   
    }
    return false;
    
}

