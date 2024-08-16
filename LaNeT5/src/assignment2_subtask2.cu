#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>
#include <float.h>

using namespace std;

// CUDA kernel for convolution with padding
__global__ void convolutionWithPaddingKernel(float* input, float* kernel, float* output,
                                             int input_rows, int input_cols, 
                                             int kernel_size, int padding) {
    int output_row = blockIdx.y * blockDim.y + threadIdx.y;
    int output_col = blockIdx.x * blockDim.x + threadIdx.x;
    // printf("Thread ");
    if (output_row < (input_rows+2*padding - kernel_size + 1) && output_col < (input_cols+2*padding - kernel_size + 1)) {
        float sum = 0.0f;
        for (int ki = 0; ki < kernel_size; ++ki) {
            for (int kj = 0; kj < kernel_size; ++kj) {
                int input_row = output_row + ki - padding;
                int input_col = output_col + kj - padding;
                if (input_row >= 0 && input_row < input_rows && input_col >= 0 && input_col < input_cols)
                    sum += input[input_row * input_cols + input_col] * kernel[ki * kernel_size + kj];
                // count++;
                // if(output_row==1 && output_col==0) {
                //     printf("input Row: %d | Input col: %d | kernel row %d | kernel Col %d | input val: %f \n",
                //     input_row, input_col, ki, kj, input[input_row * input_cols + input_col] * kernel[ki * kernel_size + kj]);
                //     printf("SUM: %f \n", sum);
                // }
                // printf("COUNT: %d \n", count);
                
            }
        }
        output[output_row * (input_cols+2*padding - kernel_size + 1) + output_col] = sum;
        // printf("OUTPUT INDEX: %d val: %f \n", (output_row * (input_cols+2*padding - kernel_size + 1) + output_col), sum);
        // cout<<"OUTPUT INDEX: "<<(output_row * (input_cols - kernel_size + 1) + output_col)<<" val: "<<sum<<endl;
    }
}

// Function to perform convolution with padding using CUDA
std::vector<vector<float>> convolutionWithPaddingGPU(const std::vector<vector<float>>& input_matrix,
                                              const std::vector<vector<float>>& kernel, 
                                              int padding) {
    int input_rows= input_matrix.size();
    int input_cols= input_matrix[0].size();
    int kernel_size= kernel.size();
    int paddedRows= input_rows+2*padding;
    int paddedCols= input_cols+2*padding;
    int output_rows = paddedRows - kernel_size + 1;
    int output_cols = paddedCols - kernel_size + 1;

    // Allocate device memory
    float *d_input, *d_kernel, *d_output;
    cudaMalloc((void**)&d_input, input_rows * input_cols * sizeof(float));
    cudaMalloc((void**)&d_kernel, kernel_size * kernel_size * sizeof(float));
    cudaMalloc((void**)&d_output, output_rows * output_cols * sizeof(float));

    float inputArray[input_rows * input_cols];
    for(int i=0; i<input_rows; i++) {
        for(int j=0; j<input_cols; j++) {
            inputArray[i*input_rows+j]= input_matrix[i][j];
            
        }
    }
    float kArray[kernel_size * kernel_size];
    for(int i=0; i<kernel_size; i++) {
        for(int j=0; j<kernel_size; j++) {
            kArray[i*kernel_size+j]= kernel[i][j];
        }
    }

    // Copy input and kernel to device memory
    cudaMemcpy(d_input, inputArray, input_rows * input_cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kArray, kernel_size * kernel_size * sizeof(float), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockSize(16, 16);  // 16x16 block size
    dim3 gridSize((input_cols + blockSize.x - 1) / blockSize.x, (input_rows + blockSize.y - 1) / blockSize.y);

    // Launch CUDA kernel
    convolutionWithPaddingKernel<<<gridSize, blockSize>>>(d_input, d_kernel, d_output,
                                                          input_rows, input_cols, kernel_size, padding);

    // Copy result back to host
    std::vector<float> output(output_rows * output_cols);
    cudaMemcpy(output.data(), d_output, output_rows * output_cols * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);

    vector<vector<float>> outputMatrix(output_rows, vector<float>(output_cols));
    for(int i=0; i<output_rows; i++) {
        for(int j=0; j<output_cols; j++) {
            outputMatrix[i][j]= output[i*output_rows+j];
        }
    }

    return outputMatrix;
}


// CUDA kernel for max pooling
__global__
void maxPoolingKernel(const float* input, float* output,
                      int inputSize, int poolSize, int outputSize) {
    int outputRow = blockIdx.y * blockDim.y + threadIdx.y;
    int outputCol = blockIdx.x * blockDim.x + threadIdx.x;

    if (outputRow < outputSize && outputCol < outputSize) {
        float maxVal = -FLT_MAX;
        for (int k = 0; k < poolSize; ++k) {
            for (int l = 0; l < poolSize; ++l) {
                int input_row = outputRow + k;
                int input_col = outputCol + l;
                if (input_row >= 0 && input_row < inputSize && input_col >= 0 && input_col < inputSize) {
                    float value = input[input_row * inputSize + input_col];
                    maxVal = fmaxf(maxVal, value);
                }
            }
        }
        output[outputRow * outputSize + outputCol] = maxVal;
    }
}

// Function to perform max pooling using CUDA
std::vector<std::vector<float>> maxPoolingGPU(const std::vector<std::vector<float>>& input, int poolSize) {
    int inputSize = input.size();
    int outputSize = inputSize-poolSize+1; // Calculate output matrix size

    // Allocate memory on GPU
    float *d_input, *d_output;
    cudaMalloc((void**)&d_input, inputSize * inputSize * sizeof(float));
    cudaMalloc((void**)&d_output, outputSize * outputSize * sizeof(float));


    float inputArray[inputSize * inputSize];
    for(int i=0; i<inputSize; i++) {
        for(int j=0; j<inputSize; j++) {
            inputArray[i*inputSize+j]= input[i][j];
        }
    }
    // Copy input data to GPU
    cudaMemcpy(d_input, inputArray, inputSize * inputSize * sizeof(float), cudaMemcpyHostToDevice);

    // Define grid and block dimensions for CUDA kernel launch
    dim3 blockDim(16, 16); // 16x16 thread block
    dim3 gridDim((inputSize + blockDim.x - 1) / blockDim.x,
                 (inputSize + blockDim.y - 1) / blockDim.y);

    // Launch CUDA kernel for max pooling
    maxPoolingKernel<<<gridDim, blockDim>>>(d_input, d_output,
                                            inputSize, poolSize, outputSize);

    // Copy output data from GPU to CPU
    std::vector<float> output(outputSize * outputSize);
    cudaMemcpy(output.data(), d_output, outputSize * outputSize * sizeof(float), cudaMemcpyDeviceToHost);

    vector<vector<float>> outputMatrix(outputSize, vector<float>(outputSize));

    for(int i=0; i<outputSize; i++) {
        for(int j=0; j<outputSize; j++) {
            outputMatrix[i][j]= output[i*outputSize+j];
        }
    }

    // Free GPU memory
    cudaFree(d_input);
    cudaFree(d_output);

    return outputMatrix;
}

// CUDA kernel for max pooling
__global__
void averagePoolingKernel(const float* input, float* output,
                      int inputSize, int poolSize, int outputSize) {
    int outputRow = blockIdx.y * blockDim.y + threadIdx.y;
    int outputCol = blockIdx.x * blockDim.x + threadIdx.x;

    if (outputRow < outputSize && outputCol < outputSize) {
        float sum = 0.0;
        for (int k = 0; k < poolSize; ++k) {
            for (int l = 0; l < poolSize; ++l) {
                int input_row = outputRow + k;
                int input_col = outputCol + l;
                if (input_row >= 0 && input_row < inputSize && input_col >= 0 && input_col < inputSize) {
                    sum += input[input_row * inputSize + input_col];
                }
            }
        }
        output[outputRow * outputSize + outputCol] = sum / (poolSize * poolSize);
    }
}

// Function to perform max pooling using CUDA
std::vector<std::vector<float>> averagePoolingGPU(const std::vector<std::vector<float>>& input, int poolSize) {
    int inputSize = input.size();
    int outputSize = inputSize-poolSize+1; // Calculate output matrix size
    // Allocate memory on GPU
    float *d_input, *d_output;
    cudaMalloc((void**)&d_input, inputSize * inputSize * sizeof(float));
    cudaMalloc((void**)&d_output, outputSize * outputSize * sizeof(float));


    float inputArray[inputSize * inputSize];
    for(int i=0; i<inputSize; i++) {
        for(int j=0; j<inputSize; j++) {
            inputArray[i*inputSize+j]= input[i][j];
        }
    }
    // Copy input data to GPU
    cudaMemcpy(d_input, inputArray, inputSize * inputSize * sizeof(float), cudaMemcpyHostToDevice);

    // Define grid and block dimensions for CUDA kernel launch
    dim3 blockDim(16, 16); // 16x16 thread block
    dim3 gridDim((inputSize + blockDim.x - 1) / blockDim.x,
                 (inputSize + blockDim.y - 1) / blockDim.y);

    // Launch CUDA kernel for max pooling
    averagePoolingKernel<<<gridDim, blockDim>>>(d_input, d_output,
                                            inputSize, poolSize, outputSize);

    // Copy output data from GPU to CPU
    std::vector<float> output(outputSize * outputSize);
    cudaMemcpy(output.data(), d_output, outputSize * outputSize * sizeof(float), cudaMemcpyDeviceToHost);

    vector<vector<float>> outputMatrix(outputSize, vector<float>(outputSize));

    for(int i=0; i<outputSize; i++) {
        for(int j=0; j<outputSize; j++) {
            outputMatrix[i][j]= output[i*outputSize+j];
        }
    }

    // Free GPU memory
    cudaFree(d_input);
    cudaFree(d_output);

    return outputMatrix;
}

// Function to apply ReLU activation function
vector<vector<float>> applyReLU(const vector<vector<float>>& input) {
    vector<vector<float>> output = input;
    for (auto& row : output) {
        for (auto& elem : row) {
            elem = max(0.0f, elem);
        }
    }
    return output;
}

// Function to apply Tanh activation function
vector<vector<float>> applyTanh(const vector<vector<float>>& input) {
    vector<vector<float>> output = input;
    for (auto& row : output) {
        for (auto& elem : row) {
            elem = tanh(elem);
        }
    }
    return output;
}

// Function to apply softmax function to a vector
vector<float> softmax(const vector<float>& input) {
    vector<float> output(input.size());
    float maxVal = *max_element(input.begin(), input.end());
    float sum = 0.0;
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = exp(input[i] - maxVal);
        sum += output[i];
    }
    for (size_t i = 0; i < output.size(); ++i) {
        output[i] /= sum;
    }
    return output;
}

// Function to apply sigmoid function to a vector
vector<float> sigmoid(const vector<float>& input) {
    vector<float> output(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = 1 / (1 + exp(-input[i]));
    }
    return output;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cout << "Insufficient arguments." << endl;
        return 1;
    }

    int task = stoi(argv[1]);
    
    switch (task) {
        case 1: {
            // Convolution
            if (argc < 6) {
                cout << "Insufficient arguments for convolution." << endl;
                return 1;
            }

            int N = stoi(argv[2]);
            int M = stoi(argv[3]);
            int padding = stoi(argv[4]);

            vector<vector<float>> input(N, vector<float>(N));
            vector<vector<float>> kernel(M, vector<float>(M));

            // Parse input and kernel matrices
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    input[i][j] = stof(argv[5 + i * N + j]);
                }
            }

            for (int i = 0; i < M; ++i) {
                for (int j = 0; j < M; ++j) {
                    kernel[i][j] = stof(argv[5 + N * N + i * M + j]);
                }
            }

            // Perform convolution
            vector<vector<float>> result = convolutionWithPaddingGPU(input, kernel, padding);

            // Output the result
            for (const auto& row : result) {
                for (const auto& val : row) {
                    cout << val << " ";
                }
                cout << endl;
            }
            break;
        }
        case 2: {
            // Non-linear activations
            if (argc < 5) {
                cout << "Insufficient arguments for non-linear activations." << endl;
                return 1;
            }

            int activation = stoi(argv[2]);
            int N = stoi(argv[3]);
            int M = stoi(argv[4]);

            vector<vector<float>> input(N, vector<float>(M));

            // Parse input matrix
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < M; ++j) {
                    input[i][j] = stof(argv[5 + i * N + j]);
                }
            }

            // Apply activation function
            vector<vector<float>> result;
            if (activation == 0) {
                result = applyReLU(input);
            } else if (activation == 1) {
                result = applyTanh(input);
            } else {
                cout << "Invalid activation function." << endl;
                return 1;
            }

            // Output the result
            for (const auto& row : result) {
                for (const auto& val : row) {
                    cout << val << " ";
                }
                cout << endl;
            }
            break;
        }
        case 3: {
            // Pooling
            if (argc < 5) {
                cout << "Insufficient arguments for pooling." << endl;
                return 1;
            }

            int poolingFunction = stoi(argv[2]);
            int poolSize = stoi(argv[3]);
            int N = stoi(argv[4]);

            vector<vector<float>> input(N, vector<float>(N));

            // Parse input matrix
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    // cout << argv[5 + i * N + j] << endl;
                    input[i][j] = stof(argv[5 + i * N + j]);
                }
            }

            // Perform pooling
            vector<vector<float>> result;
            if (poolingFunction == 0) {
                result = maxPoolingGPU(input, poolSize);
            } else if (poolingFunction == 1) {
                result = averagePoolingGPU(input, poolSize);
            } else {
                cout << "Invalid pooling function." << endl;
                return 1;
            }

            // Output the result
            for (const auto& row : result) {
                for (const auto& val : row) {
                    cout << val << " ";
                }
                cout << endl;
            }
            break;
        }
        case 4: {
            // Converting to probabilities
            if (argc < 4) {
                cout << "Insufficient arguments for converting to probabilities." << endl;
                return 1;
            }

            int function = stoi(argv[2]);
            int size = argc - 3;

            vector<float> input(size);

            // Parse input vector
            for (int i = 0; i < size; ++i) {
                input[i] = stof(argv[3 + i]);
            }

            // Perform conversion
            vector<float> result;
            if (function == 0) {
                result = sigmoid(input);
            } else if (function == 1) {
                result = softmax(input);
            } else {
                cout << "Invalid conversion function." << endl;
                return 1;
            }

            // Output the result
            for (const auto& val : result) {
                cout << val << " ";
            }
            cout << endl;
            break;
        }
        default:
            cout << "Invalid task number." << endl;
            return 1;
    }

    return 0;
}
