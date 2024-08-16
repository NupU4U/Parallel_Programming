#include <iostream>
#include <fstream>
#include <cmath>
#include <cfloat>
#include <string>
#include <dirent.h> 
#include <algorithm>
#include <ctime> 
#include <cuda_runtime.h> 
#include <device_launch_parameters.h>

inline int idx3D(int c, int i, int j, int width, int height) {
    return (c * width * height) + (i * width) + j;
}

inline int idx4D(int d, int c, int ki, int kj, int inputChannels, int kernelSize) {
    return (d * inputChannels * kernelSize * kernelSize) + (c * kernelSize * kernelSize) + (ki * kernelSize) + kj;
}
bool compareIndexedValues(const std::pair<float, int>& a, const std::pair<float, int>& b) {
    return a.first > b.first; // Sort in descending order based on value
}

__host__
void readInputArrayFromFile(float inputArray[], const std::string& filename, int inputSize) {
    // Open the input file
    std::ifstream inputFile(filename.c_str());
    if (!inputFile.is_open()) {
        std::cerr << "Error opening file!" << std::endl;
        return;
    }

    // Read the input array
    for (int i = 0; i < inputSize; ++i) {
        if (!(inputFile >> inputArray[i])) {
            std::cerr << "Error reading from file!" << std::endl;
            inputFile.close();
            return;
        }
    }

    // Close the input file
    inputFile.close();
}

__host__
void readConvolutionFromFile(float convMatrix[], float bias[], const std::string& filename, int outputChannels, int inputChannels, int kernelSize) {
    const int convMatrixSize = outputChannels * inputChannels * kernelSize * kernelSize;

    std::ifstream file(filename.c_str());
    if (!file.is_open()) {
        std::cerr << "Error opening file!\n";
        return;
    }

    for (int i = 0; i < convMatrixSize; ++i) {
        file >> convMatrix[i];
    }

    // Read bias
    for (int i = 0; i < outputChannels; ++i) {
        file >> bias[i];
    }

    file.close();
}

__global__ void convolution(const float* inputMatrix, const float* kernel, const float* bias,
                                  float* outputMatrix, int inputChannel, int outputChannel,
                                  int inputSize, int outputSize, int kernelSize) {
    // Calculate the global thread index
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int d = blockIdx.z;
 
    if (i < outputSize && j < outputSize) {
        float sum = 0.0f;
        for (int c = 0; c < inputChannel; c++) {
            for (int ki = 0; ki < kernelSize; ki++) {
                for (int kj = 0; kj < kernelSize; kj++) {
                    // int inputIdx = idx3D(c, i + ki, j + kj, inputSize, inputSize);
                    int inputIdx =(c*inputSize*inputSize)+ ((i + ki)* inputSize) + (j + kj);
                    // int kernelIdx = idx4D(d, c, ki, kj, inputChannel, kernelSize);
                    int kernelIdx=(d * inputChannel * kernelSize * kernelSize) + (c * kernelSize * kernelSize) + (ki * kernelSize) + kj;
                    sum += inputMatrix[inputIdx] * kernel[kernelIdx];
                }
            }
        }
        int outIdx= (d* outputSize*outputSize) + (i*outputSize) +j;
        // outputMatrix[idx3D(d, i, j, outputSize, outputSize)] = sum + bias[d];
        outputMatrix[outIdx] = sum + bias[d];
    }
}

void fullyConnected(const float* inputMatrix, const float* kernel, const float* bias,
                 float* outputMatrix, int inputChannel, int outputChannel, int inputSize, int outputSize, int kernelSize) {
    for (int d = 0; d < outputChannel; d++) {
        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                float sum = 0.0f;
                for (int c = 0; c < inputChannel; c++) {
                    for (int ki = 0; ki < kernelSize; ki++) {
                        for (int kj = 0; kj < kernelSize; kj++) {
                            sum += inputMatrix[idx3D(c, i + ki, j + kj, inputSize, inputSize)]
                                * kernel[idx4D(d, c, ki, kj, inputChannel, kernelSize)];
                        }
                    }
                }
                outputMatrix[idx3D(d, i, j, outputSize, outputSize)] = sum + bias[d];
            }
        }
    }
}

__global__ void maxPoolingKernel(const float* inputVolume, float* outputVolume,
                                 int inputChannels, int inputSize,
                                 int outputSize, int kernelSize) {
    // Calculate the global thread index
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z;

    if (i < outputSize && j < outputSize) {
        // Initialize the maximum value
        float maxVal = -FLT_MAX;

        // Iterate over the pooling window
        for (int ki = 0; ki < kernelSize; ++ki) {
            for (int kj = 0; kj < kernelSize; ++kj) {
                int inputIdx = c * inputSize * inputSize + (i * kernelSize + ki) * inputSize + (j * kernelSize + kj);
                float val = inputVolume[inputIdx];
                maxVal = fmaxf(maxVal, val);
            }
        }

        // Assign the maximum value to the output
        outputVolume[c * outputSize * outputSize + i * outputSize + j] = maxVal;
    }
}

void applyReLU(float* outputMatrix, int outputSize, int outputChannel) {
    for (int d = 0; d < outputChannel; ++d) {
        for (int i = 0; i < outputSize; ++i) {
            for (int j = 0; j < outputSize; ++j) {
                outputMatrix[idx3D(d, i, j, outputSize, outputSize)] = std::max(0.0f, outputMatrix[idx3D(d, i, j, outputSize, outputSize)]);
            }
        }
    }
}

void applySoftmax(float* outputMatrix,float* outputMatrixx, int outputSize, int outputChannels) {
    for (int d = 0; d < outputSize; ++d) {
        float expSum = 0.0f;
        for (int i = 0; i < outputChannels; ++i) {
            expSum += std::exp(outputMatrix[d * outputSize + i]);
        }
        for (int i = 0; i < outputChannels; ++i) {
            outputMatrixx[d * outputSize + i] = std::exp(outputMatrix[d * outputSize + i]) / expSum;
        }
    }
}

void getMaxProbabilityIndex_streams(const std::string& filenaame, float* convMatrix, float* convMatrix2, float* convMatrix3, float* convMatrix4, float* bias1,float* bias2, float* bias3, float* bias4, cudaStream_t stream, std::pair<float, int> top5[10]) {
    const int inputSizeImage = 28 * 28 * 1; // Size of the flattened array
    const int inputSize1 = 28;
    const int inputChannel1 = 1;
    const int outputChannel1 = 20;
    const int kernelSize1 = 5;
    // float inputArray[inputSizeImage]; 
    float * inputArray;
    cudaMallocHost(&inputArray, inputSizeImage * sizeof(float));
    const std::string filename2 = filenaame;
    readInputArrayFromFile(inputArray, filename2, inputSizeImage);

    const int outputSize1 = inputSize1 - kernelSize1 + 1;
    float *d_inputMatrix, *d_convMatrix, *d_bias, *d_outputMatrix;
    cudaMalloc(&d_inputMatrix, 784 * sizeof(float));
    cudaMalloc(&d_convMatrix, outputChannel1 * inputChannel1 * kernelSize1 * kernelSize1 * sizeof(float));
    cudaMalloc(&d_bias, outputChannel1 * sizeof(float));
    cudaMalloc(&d_outputMatrix, outputChannel1 * outputSize1 * outputSize1 * sizeof(float));
    // cudaMemcpy(d_inputMatrix, inputArray, inputSizeImage * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_convMatrix, convMatrix, outputChannel1 * inputChannel1 * kernelSize1 * kernelSize1 * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_bias, bias1, outputChannel1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_inputMatrix, inputArray, inputSizeImage * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_convMatrix, convMatrix, outputChannel1 * inputChannel1 * kernelSize1 * kernelSize1 * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_bias, bias1, outputChannel1 * sizeof(float), cudaMemcpyHostToDevice, stream);
    dim3 blockSize(16, 16);
    dim3 gridSize((inputSize1 + blockSize.x - 1) / blockSize.x, (inputSize1 + blockSize.y - 1) / blockSize.y, outputChannel1);
    // convolution<<<gridSize, blockSize, inputChannel1 * kernelSize1 * kernelSize1 * sizeof(float)>>>(d_inputMatrix, d_convMatrix, d_bias, d_outputMatrix, inputChannel1, outputChannel1, inputSize1, inputSize1 - kernelSize1 + 1, kernelSize1);
    convolution<<<gridSize, blockSize, 0, stream>>>(d_inputMatrix, d_convMatrix, d_bias, d_outputMatrix, inputChannel1, outputChannel1, inputSize1, inputSize1 - kernelSize1 + 1, kernelSize1);
    cudaFree(d_inputMatrix);
    cudaFree(d_convMatrix);
    cudaFree(d_bias);

    const int poolKernelSize1 = 2;
    const int poolStride1 = 2;
    const int pooledSize1 = outputSize1 / poolStride1;
    float *d_pooledOutput;
    cudaMalloc(&d_pooledOutput, outputChannel1 * pooledSize1 * pooledSize1 * sizeof(float));
    dim3 poolBlockSize(poolKernelSize1, poolKernelSize1);
    dim3 poolGridSize((pooledSize1 + poolBlockSize.x - 1) / poolBlockSize.x, (pooledSize1 + poolBlockSize.y - 1) / poolBlockSize.y, outputChannel1);
    maxPoolingKernel<<<poolGridSize, poolBlockSize,0, stream>>>(d_outputMatrix, d_pooledOutput, outputChannel1, outputSize1, pooledSize1, poolKernelSize1);
    cudaFree(d_outputMatrix);

    const int inputSize2 = pooledSize1;//12
    const int inputChannel2 = outputChannel1;//20
    const int outputChannel2 = 50;
    const int kernelSize2 = 5;
    const int outputSize2 = inputSize2 - kernelSize2 + 1;//8
    float *d_convMatrix2, *d_bias2, *d_outputMatrix2;
    cudaMalloc(&d_convMatrix2, outputChannel2 * kernelSize2 * kernelSize2 * inputChannel2 * sizeof(float));
    cudaMalloc(&d_bias2, outputChannel2 * sizeof(float));
    cudaMalloc(&d_outputMatrix2, outputChannel2 * outputSize2 * outputSize2 * sizeof(float));
    // cudaMemcpy(d_convMatrix2, convMatrix2, outputChannel2  * kernelSize2 * kernelSize2 * inputChannel2 * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_bias2, bias2, outputChannel2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_convMatrix2, convMatrix2, outputChannel2  * kernelSize2 * kernelSize2 * inputChannel2 * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_bias2, bias2, outputChannel2 * sizeof(float), cudaMemcpyHostToDevice, stream);
    dim3 blockSize2(16, 16);
    dim3 gridSize2((inputSize2 + blockSize2.x - 1) / blockSize2.x, (inputSize2 + blockSize2.y - 1) / blockSize2.y, outputChannel2);
    // convolution<<<gridSize2, blockSize2, inputChannel2 * kernelSize2 * kernelSize2 * sizeof(float)>>>(d_pooledOutput, d_convMatrix2, d_bias2, d_outputMatrix2, inputChannel2, outputChannel2, inputSize2, inputSize2 - kernelSize2 + 1, kernelSize2);
    convolution<<<gridSize2, blockSize2, 0, stream>>>(d_pooledOutput, d_convMatrix2, d_bias2, d_outputMatrix2, inputChannel2, outputChannel2, inputSize2, inputSize2 - kernelSize2 + 1, kernelSize2);
    cudaFree(d_pooledOutput);
    cudaFree(d_convMatrix2);
    cudaFree(d_bias2);

    const int poolKernelSize2 = 2;
    const int poolStride2 = 2;
    const int pooledSize2 = outputSize2 / poolStride2; //4
    float *d_pooledOutput2;
    cudaMalloc(&d_pooledOutput2, outputChannel2 * pooledSize2 * pooledSize2 * sizeof(float));
    dim3 poolBlockSize2(poolKernelSize2, poolKernelSize2);
    dim3 poolGridSize2((pooledSize2 + poolBlockSize2.x - 1) / poolBlockSize2.x, (pooledSize2 + poolBlockSize2.y - 1) / poolBlockSize2.y, outputChannel2);
    // maxPoolingKernel<<<poolGridSize2, poolBlockSize2,0>>>(d_outputMatrix2, d_pooledOutput2, outputChannel2, outputSize2, pooledSize2, poolKernelSize2);
    maxPoolingKernel<<<poolGridSize2, poolBlockSize2,0, stream>>>(d_outputMatrix2, d_pooledOutput2, outputChannel2, outputSize2, pooledSize2, poolKernelSize2);
    cudaFree(d_outputMatrix2);

    const int inputSize3 = pooledSize2; //4
    const int inputChannel3 = outputChannel2;   //50
    const int outputChannel3 = 500; 
    const int kernelSize3 = 4;
    const int outputSize3 = inputSize3 - kernelSize3 + 1; //1
    // float outputMatrix3[outputChannel3 * outputSize3 * outputSize3]; //500*1*1
    float *outputMatrix3;
    cudaMallocHost(&outputMatrix3, outputChannel3 * outputSize3 * outputSize3 * sizeof(float));
    float *d_convMatrix3, *d_bias3, *d_outputMatrix3;
    cudaMalloc(&d_convMatrix3, outputChannel3 * kernelSize3 * kernelSize3 * inputChannel3 * sizeof(float));
    cudaMalloc(&d_bias3, outputChannel3 * sizeof(float));
    cudaMalloc(&d_outputMatrix3, outputChannel3 * outputSize3 * outputSize3 * sizeof(float));
    // cudaMemcpy(d_convMatrix3, convMatrix3, outputChannel3  * kernelSize3 * kernelSize3 * inputChannel3 * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_bias3, bias3, outputChannel3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_convMatrix3, convMatrix3, outputChannel3  * kernelSize3 * kernelSize3 * inputChannel3 * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_bias3, bias3, outputChannel3 * sizeof(float), cudaMemcpyHostToDevice, stream);
    dim3 blockSize3(16, 16);
    dim3 gridSize3((inputSize3 + blockSize3.x - 1) / blockSize3.x, (inputSize3 + blockSize3.y - 1) / blockSize3.y, outputChannel3);
    // convolution<<<gridSize3, blockSize3, inputChannel3 * kernelSize3 * kernelSize3 * sizeof(float)>>>(d_pooledOutput2, d_convMatrix3, d_bias3, d_outputMatrix3, inputChannel3, outputChannel3, inputSize3, inputSize3 - kernelSize3 + 1, kernelSize3);
    convolution<<<gridSize3, blockSize3, 0, stream>>>(d_pooledOutput2, d_convMatrix3, d_bias3, d_outputMatrix3, inputChannel3, outputChannel3, inputSize3, inputSize3 - kernelSize3 + 1, kernelSize3);
    cudaMemcpy(outputMatrix3, d_outputMatrix3, outputChannel3 * (inputSize3 - kernelSize3 + 1) * (inputSize3 - kernelSize3 + 1) * sizeof(float), cudaMemcpyDeviceToHost);    
    cudaFree(d_pooledOutput2);
    cudaFree(d_convMatrix3);
    cudaFree(d_bias3);
    cudaFree(d_outputMatrix3);

    applyReLU(outputMatrix3, outputSize3, outputChannel3);  //500*1*1

    const int inputSize4 = outputSize3; // 1
    const int inputChannel4 = outputChannel3; // 500
    const int outputChannel4 = 10;  // 10
    const int kernelSize4 = 1;
    const int outputSize4 = inputSize4 - kernelSize4 + 1; // 1
    // float outputMatrix4[outputChannel4 * outputSize4 * outputSize4]; // 10*1*1
    float *outputMatrix4;
    cudaMallocHost(&outputMatrix4, outputChannel4 * outputSize4 * outputSize4 * sizeof(float));
    fullyConnected(outputMatrix3, convMatrix4, bias4, outputMatrix4, inputChannel4, outputChannel4, inputSize4, outputSize4, kernelSize4);

    // float outputMatrix5[outputChannel4 * outputSize4];
    float *outputMatrix5;
    cudaMallocHost(&outputMatrix5, outputChannel4 * outputSize4 * sizeof(float));
    applySoftmax(outputMatrix4, outputMatrix5, outputSize4, outputChannel4);

    //priting 
    // for (int i = 0; i < outputChannel4; ++i) {
    //     std::cout << outputMatrix5[i] << " "<< std::endl;
    // }
    // float max = -FLT_MAX;
    // int maxIndex = -1;
    // for (int i = 0; i < outputChannel4; ++i) {
    //     if (outputMatrix5[i] > max) {
    //         max = outputMatrix5[i];
    //         maxIndex = i;
    //     }
    // }
    for (int i = 0; i < outputChannel4; i++) {
        top5[i] = std::make_pair(outputMatrix5[i], i);
    }
    std::sort(top5, top5 + outputChannel4, compareIndexedValues);
    cudaFreeHost(inputArray);
    cudaFreeHost(outputMatrix3);
    cudaFreeHost(outputMatrix4);
    cudaFreeHost(outputMatrix5);
    // return maxIndex;

}
int getLastDigitOfFile(const std::string& filename) {
    // Iterate through the filename characters from right to left
   for (int i = filename.size() - 1; i >= 0; --i) {
       if (isdigit(filename[i])) { // Check if the character is a digit
           return filename[i] - '0'; // Convert character to integer and return
       }
   }
   // If no digit is found, return -1 or any default value as per your requirement
   return -1;
}
void withStreams(){ 
    // int start_s=clock();
    std::string folderPath = "pre-proc-img/";
    std::string destinationFolder = "output/";
    // int count =0;
    int grandcout = 0; 
    const int inputChannel1 = 1;
    const int outputChannel1 = 20;
    const int kernelSize1 = 5;
    // float convMatrix[outputChannel1 * inputChannel1 * kernelSize1 * kernelSize1];
    // float bias1[outputChannel1];
    float *convMatrix, *bias1;
    //storing 500*sizeof(float) in host memory
    int k = 500 * sizeof(float);
    cudaMallocHost(&convMatrix, k);
    cudaMallocHost(&bias1, outputChannel1 * sizeof(float));
    const std::string filename = "weights/conv1.txt";

    readConvolutionFromFile(convMatrix, bias1, filename, outputChannel1, inputChannel1, kernelSize1);
    // const int inputSize2 = 12;//12
    const int inputChannel2 = 20;//20
    const int outputChannel2 = 50;
    const int kernelSize2 = 5;
    // float convMatrix2[outputChannel2  * kernelSize2 * kernelSize2 * inputChannel2];//50*5*5*20
    // float bias2[outputChannel2]; //50
    float *convMatrix2, *bias2;
    cudaMallocHost(&convMatrix2, outputChannel2 * kernelSize2 * kernelSize2 * inputChannel2 * sizeof(float));
    cudaMallocHost(&bias2, outputChannel2 * sizeof(float));
    const std::string filename3 = "weights/conv2.txt";

    // Read convolution matrix and bias from file
    readConvolutionFromFile(convMatrix2, bias2, filename3, outputChannel2, inputChannel2, kernelSize2);
    // const int inputSize3 = 4; //4
    const int inputChannel3 = 50;   //50
    const int outputChannel3 = 500; 
    const int kernelSize3 = 4;
    // float convMatrix3[outputChannel3  * kernelSize3 * kernelSize3 * inputChannel3];  //500*4*4*50
    // float bias3[outputChannel3]; //500
    float *convMatrix3, *bias3;
    cudaMallocHost(&convMatrix3, outputChannel3 * kernelSize3 * kernelSize3 * inputChannel3 * sizeof(float));
    cudaMallocHost(&bias3, outputChannel3 * sizeof(float));
    const std::string filename4 = "weights/fc1.txt";

    // Read convolution matrix and bias from file
    readConvolutionFromFile(convMatrix3, bias3, filename4, outputChannel3, inputChannel3, kernelSize3);
    // const int inputSize4 = 1; // 1
    const int inputChannel4 = outputChannel3; // 500
    const int outputChannel4 = 10;  // 10
    const int kernelSize4 = 1;
    // float convMatrix4[outputChannel4  * kernelSize4 * kernelSize4 * inputChannel4]; // 10*1*1*500
    // float bias4[outputChannel4]; // 10
    float *convMatrix4, *bias4;
    cudaMallocHost(&convMatrix4, outputChannel4 * kernelSize4 * kernelSize4 * inputChannel4 * sizeof(float));
    cudaMallocHost(&bias4, outputChannel4 * sizeof(float));
    const std::string filename5 = "weights/fc2.txt";

    // Read convolution matrix and bias from file
    readConvolutionFromFile(convMatrix4, bias4, filename5, outputChannel4, inputChannel4, kernelSize4);
    int number_of_streams = 1000;
    //creating array of cuda streams 
    cudaStream_t streams[number_of_streams];
    for(int i=0;i<number_of_streams;i++){
        cudaStreamCreate(&streams[i]);
    }
    // Iterate through the folder
    DIR *dir;
    struct dirent *entry;
    if ((dir = opendir(folderPath.c_str())) != NULL) {
        while ((entry = readdir(dir)) != NULL) {
            if (entry->d_type == DT_REG) { // Check if it's a regular file
                std::string filename = entry->d_name;
                std::ifstream file((folderPath + "/" + filename).c_str());

                if (file.is_open()) {
                    // Read content from the file
                    std::pair<float, int> top5ValuesWithIndices[10];
                    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

                    // Apply your function to the content
                    getMaxProbabilityIndex_streams(folderPath + "/" + filename,  convMatrix, convMatrix2, convMatrix3, convMatrix4, bias1, bias2, bias3, bias4,streams[grandcout %number_of_streams],top5ValuesWithIndices);
                    //store the result in a file with same name in destination folder
                    std::ofstream output((destinationFolder + filename).c_str());
                    if (output.is_open()) {
                        for (int i = 0; i < 5; i++) {
                            // output << top5ValuesWithIndices[i].second << " " << top5ValuesWithIndices[i].first << std::endl;
                            //format 99.9426 class 2
                            output << top5ValuesWithIndices[i].first << " class " << top5ValuesWithIndices[i].second << std::endl;
                        }
                        // std::cout << "Grandcout: " << grandcout <<std::endl;

                        output.close();
                    } else {
                        std::cerr << "Unable to open file: " << destinationFolder + filename << std::endl;
                    }
                    grandcout++;
                } else {
                    std::cerr << "Unable to open file: " << filename << std::endl;
                }
            }
        }
        closedir(dir);
    } else {
        std::cerr << "Error: Could not open the directory!" << std::endl;
    }
    for (int i = 0; i < number_of_streams; i++) {
        cudaStreamSynchronize(streams[i]);
    }
    for(int i=0;i<number_of_streams;i++){
        cudaStreamDestroy(streams[i]);
    }
    cudaFreeHost(convMatrix);
    cudaFreeHost(bias1);
    cudaFreeHost(convMatrix2);
    cudaFreeHost(bias2);
    cudaFreeHost(convMatrix3);
    cudaFreeHost(bias3);
    cudaFreeHost(convMatrix4);
    cudaFreeHost(bias4);
//     int stop_s=clock();
//     std::cout << "Time: " << (stop_s-start_s)/double(CLOCKS_PER_SEC)*1000 << " ms" << std::endl;
//     std::cout << "Total Files: " << grandcout << std::endl;
}


void getMaxProbabilityIndex_no_streams(const std::string& filenaame, float* convMatrix, float* convMatrix2, float* convMatrix3, float* convMatrix4, float* bias1,float* bias2, float* bias3, float* bias4, std::pair<float, int>* top5){
    const int inputSizeImage = 28 * 28 * 1; // Size of the flattened array
    const int inputSize1 = 28;
    const int inputChannel1 = 1;
    const int outputChannel1 = 20;
    const int kernelSize1 = 5;
    float inputArray[inputSizeImage]; 
    const std::string filename2 = filenaame;
    readInputArrayFromFile(inputArray, filename2, inputSizeImage);

    const int outputSize1 = inputSize1 - kernelSize1 + 1;
    float *d_inputMatrix, *d_convMatrix, *d_bias, *d_outputMatrix;
    cudaMalloc(&d_inputMatrix, 784 * sizeof(float));
    cudaMalloc(&d_convMatrix, outputChannel1 * inputChannel1 * kernelSize1 * kernelSize1 * sizeof(float));
    cudaMalloc(&d_bias, outputChannel1 * sizeof(float));
    cudaMalloc(&d_outputMatrix, outputChannel1 * outputSize1 * outputSize1 * sizeof(float));
    cudaMemcpy(d_inputMatrix, inputArray, inputSizeImage * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_convMatrix, convMatrix, outputChannel1 * inputChannel1 * kernelSize1 * kernelSize1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias1, outputChannel1 * sizeof(float), cudaMemcpyHostToDevice);
    dim3 blockSize(16, 16);
    dim3 gridSize((inputSize1 + blockSize.x - 1) / blockSize.x, (inputSize1 + blockSize.y - 1) / blockSize.y, outputChannel1);
    convolution<<<gridSize, blockSize, 0>>>(d_inputMatrix, d_convMatrix, d_bias, d_outputMatrix, inputChannel1, outputChannel1, inputSize1, inputSize1 - kernelSize1 + 1, kernelSize1);
    cudaFree(d_inputMatrix);
    cudaFree(d_convMatrix);
    cudaFree(d_bias);

    const int poolKernelSize1 = 2;
    const int poolStride1 = 2;
    const int pooledSize1 = outputSize1 / poolStride1;
    float *d_pooledOutput;
    cudaMalloc(&d_pooledOutput, outputChannel1 * pooledSize1 * pooledSize1 * sizeof(float));
    dim3 poolBlockSize(poolKernelSize1, poolKernelSize1);
    dim3 poolGridSize((pooledSize1 + poolBlockSize.x - 1) / poolBlockSize.x, (pooledSize1 + poolBlockSize.y - 1) / poolBlockSize.y, outputChannel1);
    maxPoolingKernel<<<poolGridSize, poolBlockSize,0>>>(d_outputMatrix, d_pooledOutput, outputChannel1, outputSize1, pooledSize1, poolKernelSize1);
    cudaFree(d_outputMatrix);

    const int inputSize2 = pooledSize1;//12
    const int inputChannel2 = outputChannel1;//20
    const int outputChannel2 = 50;
    const int kernelSize2 = 5;
    const int outputSize2 = inputSize2 - kernelSize2 + 1;//8
    float *d_convMatrix2, *d_bias2, *d_outputMatrix2;
    cudaMalloc(&d_convMatrix2, outputChannel2 * kernelSize2 * kernelSize2 * inputChannel2 * sizeof(float));
    cudaMalloc(&d_bias2, outputChannel2 * sizeof(float));
    cudaMalloc(&d_outputMatrix2, outputChannel2 * outputSize2 * outputSize2 * sizeof(float));
    cudaMemcpy(d_convMatrix2, convMatrix2, outputChannel2  * kernelSize2 * kernelSize2 * inputChannel2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias2, bias2, outputChannel2 * sizeof(float), cudaMemcpyHostToDevice);
    dim3 blockSize2(16, 16);
    dim3 gridSize2((inputSize2 + blockSize2.x - 1) / blockSize2.x, (inputSize2 + blockSize2.y - 1) / blockSize2.y, outputChannel2);
    convolution<<<gridSize2, blockSize2, 0>>>(d_pooledOutput, d_convMatrix2, d_bias2, d_outputMatrix2, inputChannel2, outputChannel2, inputSize2, inputSize2 - kernelSize2 + 1, kernelSize2);
    cudaFree(d_pooledOutput);
    cudaFree(d_convMatrix2);
    cudaFree(d_bias2);

    const int poolKernelSize2 = 2;
    const int poolStride2 = 2;
    const int pooledSize2 = outputSize2 / poolStride2; //4
    float *d_pooledOutput2;
    cudaMalloc(&d_pooledOutput2, outputChannel2 * pooledSize2 * pooledSize2 * sizeof(float));
    dim3 poolBlockSize2(poolKernelSize2, poolKernelSize2);
    dim3 poolGridSize2((pooledSize2 + poolBlockSize2.x - 1) / poolBlockSize2.x, (pooledSize2 + poolBlockSize2.y - 1) / poolBlockSize2.y, outputChannel2);
    maxPoolingKernel<<<poolGridSize2, poolBlockSize2,0>>>(d_outputMatrix2, d_pooledOutput2, outputChannel2, outputSize2, pooledSize2, poolKernelSize2);
    cudaFree(d_outputMatrix2);

    const int inputSize3 = pooledSize2; //4
    const int inputChannel3 = outputChannel2;   //50
    const int outputChannel3 = 500; 
    const int kernelSize3 = 4;
    const int outputSize3 = inputSize3 - kernelSize3 + 1; //1
    float outputMatrix3[outputChannel3 * outputSize3 * outputSize3]; //500*1*1
    float *d_convMatrix3, *d_bias3, *d_outputMatrix3;
    cudaMalloc(&d_convMatrix3, outputChannel3 * kernelSize3 * kernelSize3 * inputChannel3 * sizeof(float));
    cudaMalloc(&d_bias3, outputChannel3 * sizeof(float));
    cudaMalloc(&d_outputMatrix3, outputChannel3 * outputSize3 * outputSize3 * sizeof(float));
    cudaMemcpy(d_convMatrix3, convMatrix3, outputChannel3  * kernelSize3 * kernelSize3 * inputChannel3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias3, bias3, outputChannel3 * sizeof(float), cudaMemcpyHostToDevice);
    dim3 blockSize3(16, 16);
    dim3 gridSize3((inputSize3 + blockSize3.x - 1) / blockSize3.x, (inputSize3 + blockSize3.y - 1) / blockSize3.y, outputChannel3);
    convolution<<<gridSize3, blockSize3, 0>>>(d_pooledOutput2, d_convMatrix3, d_bias3, d_outputMatrix3, inputChannel3, outputChannel3, inputSize3, inputSize3 - kernelSize3 + 1, kernelSize3);
    cudaMemcpy(outputMatrix3, d_outputMatrix3, outputChannel3 * (inputSize3 - kernelSize3 + 1) * (inputSize3 - kernelSize3 + 1) * sizeof(float), cudaMemcpyDeviceToHost);    
    cudaFree(d_pooledOutput2);
    cudaFree(d_convMatrix3);
    cudaFree(d_bias3);
    cudaFree(d_outputMatrix3);

    applyReLU(outputMatrix3, outputSize3, outputChannel3);  //500*1*1

    const int inputSize4 = outputSize3; // 1
    const int inputChannel4 = outputChannel3; // 500
    const int outputChannel4 = 10;  // 10
    const int kernelSize4 = 1;
    const int outputSize4 = inputSize4 - kernelSize4 + 1; // 1
    float outputMatrix4[outputChannel4 * outputSize4 * outputSize4]; // 10*1*1
    fullyConnected(outputMatrix3, convMatrix4, bias4, outputMatrix4, inputChannel4, outputChannel4, inputSize4, outputSize4, kernelSize4);
    float outputMatrix5[outputChannel4 * outputSize4];
    applySoftmax(outputMatrix4, outputMatrix5, outputSize4, outputChannel4);

    // std::cout << "Time: " << (stop_s-start_s)/double(CLOCKS_PER_SEC)*1000 << " ms" << std::endl;
    // std::cout << "Classification Probabilities:\n";
    // for (int i = 0; i < outputChannel4; ++i) {
    //     std::cout << "Class " << i << ": " << outputMatrix5[i]<<"\n";
    // }
    //storing top 5 probabilities
    for (int i = 0; i < outputChannel4; i++) {
        top5[i] = std::make_pair(outputMatrix5[i], i);
    }
    std::sort(top5, top5 + outputChannel4, compareIndexedValues);

    // for (int i = 0; i < 5; i++) {
    //     std::cout << "Top 5: " << top5[i] << "\n";
    // }

}
void without_streams(){
    // int start_s=clock();

    // files to read in results
    std::string folderPath = "pre-proc-img/";
    DIR *dir;
    struct dirent *entry;   
    // int count = 0;
    int grandcout = 0;
    const int inputChannel1 = 1;
    const int outputChannel1 = 20;
    const int kernelSize1 = 5;
    float convMatrix[outputChannel1 * inputChannel1 * kernelSize1 * kernelSize1];
    float bias1[outputChannel1];
    const std::string filename = "weights/conv1.txt";
    // Read convolution matrix and bias from file

    readConvolutionFromFile(convMatrix, bias1, filename, outputChannel1, inputChannel1, kernelSize1);
    // const int inputSize2 = 12;//12
    const int inputChannel2 = 20;//20
    const int outputChannel2 = 50;
    const int kernelSize2 = 5;
    float convMatrix2[outputChannel2  * kernelSize2 * kernelSize2 * inputChannel2];//50*5*5*20
    float bias2[outputChannel2]; //50
    const std::string filename3 = "weights/conv2.txt";

    // Read convolution matrix and bias from file
    readConvolutionFromFile(convMatrix2, bias2, filename3, outputChannel2, inputChannel2, kernelSize2);
    // const int inputSize3 = 4; //4
    const int inputChannel3 = 50;   //50
    const int outputChannel3 = 500; 
    const int kernelSize3 = 4;
    float convMatrix3[outputChannel3  * kernelSize3 * kernelSize3 * inputChannel3];  //500*4*4*50
    float bias3[outputChannel3]; //500
    const std::string filename4 = "weights/fc1.txt";

    // Read convolution matrix and bias from file
    readConvolutionFromFile(convMatrix3, bias3, filename4, outputChannel3, inputChannel3, kernelSize3);
    // const int inputSize4 = 1; // 1
    const int inputChannel4 = outputChannel3; // 500
    const int outputChannel4 = 10;  // 10
    const int kernelSize4 = 1;
    float convMatrix4[outputChannel4  * kernelSize4 * kernelSize4 * inputChannel4]; // 10*1*1*500
    float bias4[outputChannel4]; // 10
    const std::string filename5 = "weights/fc2.txt";

    // Read convolution matrix and bias from file
    readConvolutionFromFile(convMatrix4, bias4, filename5, outputChannel4, inputChannel4, kernelSize4);
    const std::string destinationFolder = "output/";

    if ((dir = opendir(folderPath.c_str())) != NULL) {
        while ((entry = readdir(dir)) != NULL) {

            if (entry->d_type == DT_REG) { // Check if it's a regular file
                std::string filename = entry->d_name;
                std::ifstream file((folderPath + "/" + filename).c_str());

                if (file.is_open()) {
                    // Read content from the file
                    std::pair<float, int> top5ValuesWithIndices[10];
                    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

                    // Apply your function to the content
                    getMaxProbabilityIndex_no_streams(folderPath + "/" + filename,  convMatrix, convMatrix2, convMatrix3, convMatrix4, bias1, bias2, bias3, bias4,top5ValuesWithIndices);
                    //store the result in a file with same name in destination folder
                    std::ofstream output((destinationFolder + filename).c_str());
                    if (output.is_open()) {
                        for (int i = 0; i < 5; i++) {
                            // output << top5ValuesWithIndices[i].second << " " << top5ValuesWithIndices[i].first << std::endl;
                            //format 99.9426 class 2
                            output << top5ValuesWithIndices[i].first << " class " << top5ValuesWithIndices[i].second << std::endl;
                        }
                        // std::cout << "Grandcout: " << grandcout <<std::endl;
                        output.close();
                    } else {
                        std::cerr << "Unable to open file: " << destinationFolder + filename << std::endl;
                    }
                    grandcout++;
                } else {
                    std::cerr << "Unable to open file: " << filename << std::endl;
                }
            }
        }
        closedir(dir);
    } else {
        // Could not open directory
        std::cerr << "Error opening directory: " << folderPath << std::endl;
    }  
    // std::cout << "Total files with correct results: " << count << std::endl;
    // std::cout << "Total files processed: " << grandcout << std::endl;
    // int stop_s=clock();
    // std::cout << "Time: " << (stop_s-start_s)/double(CLOCKS_PER_SEC)*1000 << " ms" << std::endl;
}
int main(int argc, char* argv[]){
    // ./subtask4 [1 - with streams, 0 - without streams] 
    int choice = 0;
    if ( argc <1){
        std::cerr << "Please provide the choice of execution" << std::endl;
        return 1;
    }
    choice = atoi(argv[1]);
    if (choice == 1){
        withStreams();
    } else if (choice == 0){
        without_streams();
    } else {
        std::cerr << "Invalid choice" << std::endl;
    }
    
}