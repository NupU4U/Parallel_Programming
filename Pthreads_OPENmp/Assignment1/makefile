# Makefile for compiling openMP_ass1.cpp and pthread_ass1.cpp

# Compiler
CC = g++

# Compiler flags
CFLAGS = -Wall -Wextra -pedantic -std=c++11
OMPFLAGS = -fopenmp
PTHREADFLAGS = -pthread

# Source files
OPENMP_SRC = openMP_ass1.cpp
PTHREAD_SRC = pthread_ass1.cpp

# Output executables
OPENMP_OUT = openMP_ass1
PTHREAD_OUT = pthread_ass1

all: openMP pthread

openMP: $(OPENMP_SRC)
	$(CC) $(CFLAGS) $(OMPFLAGS) $(OPENMP_SRC) -o $(OPENMP_OUT)

pthread: $(PTHREAD_SRC)
	$(CC) $(CFLAGS) $(PTHREADFLAGS) $(PTHREAD_SRC) -o $(PTHREAD_OUT)

clean:
	rm -f $(OPENMP_OUT) $(PTHREAD_OUT)
