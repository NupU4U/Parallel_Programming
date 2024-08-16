#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace std;

// Function to perform convolution with padding
vector<vector<float>> convolutionWithPadding(const vector<vector<float>>& input_matrix, const vector<vector<float>>& kernel, int padding) {
    int input_rows = input_matrix.size();
    int input_cols = input_matrix[0].size();
    int kernel_size = kernel.size();
    int padded_rows = input_rows + 2 * padding;
    int padded_cols = input_cols + 2 * padding;
    int output_rows = padded_rows - kernel_size + 1;
    int output_cols = padded_cols - kernel_size + 1;

    vector<vector<float>> padded_input(padded_rows, vector<float>(padded_cols, 0.0f));

    // Copy input_matrix to padded_input with specified padding
    for (int i = 0; i < input_rows; ++i) {
        for (int j = 0; j < input_cols; ++j) {
            padded_input[i + padding][j + padding] = input_matrix[i][j];
        }
    }

    vector<vector<float>> output(output_rows, vector<float>(output_cols, 0.0f));

    // Perform convolution
    for (int i = 0; i < output_rows; ++i) {
        for (int j = 0; j < output_cols; ++j) {
            float sum = 0.0f;
            for (int ki = 0; ki < kernel_size; ++ki) {
                for (int kj = 0; kj < kernel_size; ++kj) {
                    sum += padded_input[i + ki][j + kj] * kernel[ki][kj];
                }
            }
            output[i][j] = sum;
        }
    }
    return output;
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

// Function to perform max pooling
vector<vector<float>> maxPooling(const vector<vector<float>>& input, int poolSize) {
    int inputSize = input.size();
    int outputSize = (inputSize -(poolSize))+1;

    vector<vector<float>> output(outputSize, vector<float>(outputSize, 0.0));

    for (int i = 0; i < outputSize; ++i) {
        for (int j = 0; j < outputSize; ++j) {
            float maxVal = numeric_limits<float>::lowest();
            for (int k = 0; k < poolSize; ++k) {
                for (int l = 0; l < poolSize; ++l) {
                    int row = i + k;
                    int col = j + l;
                    if (row < inputSize && col < inputSize) {
                        maxVal = max(maxVal, input[row][col]);
                    }
                }
            }
            // cout << maxVal << endl;
            output[i][j] = maxVal;
        }
    }
    return output;
}

// Function to perform average pooling
vector<vector<float>> averagePooling(const vector<vector<float>>& input, int poolSize) {
    int inputSize = input.size();
    int outputSize = (inputSize -(poolSize))+1;

    vector<vector<float>> output(outputSize, vector<float>(outputSize, 0.0));

    for (int i = 0; i < outputSize; ++i) {
        for (int j = 0; j < outputSize; ++j) {
            float sum = 0.0;
            for (int k = 0; k < poolSize; ++k) {
                for (int l = 0; l < poolSize; ++l) {
                    sum += input[i  + k][j  + l];
                }
            }
            output[i][j] = sum / (poolSize * poolSize);
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
            vector<vector<float>> result = convolutionWithPadding(input, kernel, padding);

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
            // cout << N << " " << M << endl;
            vector<vector<float>> input(N, vector<float>(M));
            // Parse input matrix
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < M; ++j) {
                    input[i][j] = stof(argv[5 + i * M + j]);

                }
            }
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
                result = maxPooling(input, poolSize); // Assuming max pooling with pool size 2
            } else if (poolingFunction == 1) {
                result = averagePooling(input, poolSize); // Assuming average pooling with pool size 2
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

