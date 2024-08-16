# Define the C++ compiler
CXX = g++

# Define the CUDA compiler
NVCC = nvcc

# Define any compile-time flags for C++
CXXFLAGS = -std=c++11

# Define any compile-time flags for CUDA
NVCCFLAGS = -O2

# Define any directories containing header files
INCLUDES = -I.

# Output directory
OUTDIR = output

# Subtask 1 files and executable
CPP_SRCS1 = src/assignment2_subtask1.cpp
MAIN1 = subtask1

# Subtask 2 files and executable
CUDA_SRCS2 = src/assignment2_subtask2.cu 
MAIN2 = subtask2

# Subtask 3 files and executable
CUDA_SRCS3 = src/assignment2_subtask3.cu 
MAIN3 = subtask3

# Subtask 4 files and executable
CUDA_SRCS4 = src/assignment2_subtask4.cu 
MAIN4 = subtask4

run_python_script:
	python3 preprocessing.py

.PHONY: clean all subtask1 subtask2 subtask3 subtask4

all: subtask1 subtask2 subtask3 subtask4

subtask1: $(MAIN1)
$(MAIN1):
	$(CXX) $(CXXFLAGS) $(CPP_SRCS1) -o $(MAIN1)

subtask2: $(MAIN2)
$(MAIN2):
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) $(CUDA_SRCS2) -o $(MAIN2)

subtask3: $(MAIN3) run_python_script
$(MAIN3):
	$(NVCC) $(NVCCFLAGS) $(CUDA_SRCS3) -o $(MAIN3)

subtask4: $(MAIN4) run_python_script
$(MAIN4):
	$(NVCC) $(NVCCFLAGS) $(CUDA_SRCS4) -o $(MAIN4)

clean:
	rm -f $(MAIN1) $(MAIN2) $(MAIN3) $(MAIN4)

