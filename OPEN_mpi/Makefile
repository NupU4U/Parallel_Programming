# Compiler and compiler flags
CXX = mpic++
CXXFLAGS = -std=c++11 -Wall -Wextra

# Directories for source files
SRC_DIR = src/generate src/solver
BUILD_DIR = build
OBJ_DIR = $(BUILD_DIR)/obj

# Source files (list all .cpp files)
SRC = $(wildcard $(addsuffix /*.cpp,$(SRC_DIR)))
MAIN_SRC = src/maze.cpp

# Object files (generated based on source files)
OBJ = $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(notdir $(SRC)))
MAIN_OBJ = $(OBJ_DIR)/$(notdir $(MAIN_SRC:.cpp=.o))

# Output executable
TARGET = maze.out

# Main target
all: $(BUILD_DIR) $(TARGET)

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)
	mkdir -p $(OBJ_DIR)

# Compile each source file into object files
$(OBJ_DIR)/%.o: src/generate/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: src/solver/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(MAIN_OBJ): $(MAIN_SRC)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Link all object files into the executable
$(TARGET): $(OBJ) $(MAIN_OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@

# Clean build files
clean:
	rm -rf $(BUILD_DIR)

# Phony targets (do not create files with these names)
.PHONY: all clean

