#!/bin/bash

# Define sizes and thread counts
sizes=(500 1000 2000)
threads=(1 2 4 8 16)

# Iterate over sizes and thread counts
for size in "${sizes[@]}"; do
    for thread_count in "${threads[@]}"; do
        # Run the pthreads program
        ./openMP_ass1 $size $thread_count 0
    done
done

for size in "${sizes[@]}"; do
    for thread_count in "${threads[@]}"; do
        # Run the openMP program
        ./pthread_ass1 $size $thread_count 0
    done
done
