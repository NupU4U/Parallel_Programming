# Assignment-2: LENET-5 Neural Network Implementation

This is Assignment-2 for the course COL380 where the LENET-5 neural network is implemented using various functionalities including convolution, max pooling, fully connected layers, ReLU, and softmax functions. Below are the details of the implementation.

## Project Structure

The project directory structure is organized as follows:
The code submission includes the following files:
kerbno1_kerbno2_kerbno3/
├── README.md
├── img/ (images go here)
├── weights/ 
├── pre-proc-img/ (data is stored here after preprocessing)
├── preprocessing.py/preprocessing.cpp/preprocessing.cc [pre-processing source code]
├── report.pdf
├── output/ 
└── src/
    ├── assignment2_subtask1.cpp
    ├── assignment2_subtask2.cu
    ├── assignment2_subtask3.cu
    └── assignment2_subtask4.cu

The weights folder contains the weights of the network after training on the MNIST dataset. The pre-proc-img folder
contains the pre-processed images in the form of txt files. The output folder contains the output of the code on the
MNIST dataset. The src folder contains the source code for the 4 subtasks. The report.pdf contains the report of the
assignment.

## Executable Files

Executable files are generated using `make all` command. The executables are named as follows:

1. subtask1
2. subtask2
3. subtask3
4. subtask4

## Instructions for Running Executables

### Subtask 1:

1. **Convolution:** 
  ```
  ./subtask1 1 <N> <M> <P> <Matrix of size NN> <Kernel of size MM>
  ```
2. **Non-linear activations:**
- ReLU:
  ```
  ./subtask1 2 0 <N> <M> <Matrix of size N*M>
  ```
- tanH:
  ```
  ./subtask1 2 1 <N> <M> <Matrix of size N*M>
  ```

3. **Subsampling:**
- Max Pooling:
  ```
  ./subtask1 3 0 <M> <N> <Matrix of size N*N>
  ```
- Avg Pooling:
  ```
  ./subtask1 3 1 <M> <N> <Matrix of size N*N>
  ```

4. **Converting a vector:**
- Sigmoid:
  ```
  ./subtask1 4 0 <Vector of numbers>
  ```
- Softmax:
  ```
  ./subtask1 4 1 <Vector of numbers>
  ```

### Subtask 2:

Instructions for Subtask 2 are similar to Subtask 1 with the executable name `subtask2`.

### Subtask 3:

1. Run:
    ```
    ./subtask3 
    ```


### Subtask 4:

1. **With streams:**
    ```
    ./subtask4 0 
    ```
2. **Without streams:**
    ```
    ./subtask4 1 
    ```

## Conclusion

The code submission includes various functionalities and implementations related to the LENET-5 neural network architecture. The achieved accuracy on the MNIST dataset is 99.09%, which is commendable.
