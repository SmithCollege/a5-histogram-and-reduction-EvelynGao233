#include <stdio.h>
#include <cuda.h>

__global__ void gpuReductionKernel(float *input, float *output, int n) {
    extern __shared__ float sharedData[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // Load input into shared memory (each thread loads two elements)
    sharedData[tid] = (i < n) ? input[i] + input[i + blockDim.x] : 0;
    __syncthreads();

    // Perform reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sharedData[tid] += sharedData[tid + s];
        }
        __syncthreads();
    }

    // Write result of this block to output
    if (tid == 0) output[blockIdx.x] = sharedData[0];
}

void gpuReduction(float *d_input, float *d_output, int n) {
    int remainingElements = n;
    float *input = d_input;
    float *output = d_output;

    dim3 blockSize(256);
    int sharedMemSize = blockSize.x * sizeof(float);

    while (remainingElements > 1) {
        int gridSize = (remainingElements + blockSize.x * 2 - 1) / (blockSize.x * 2);

        gpuReductionKernel<<<gridSize, blockSize, sharedMemSize>>>(input, output, remainingElements);
        cudaDeviceSynchronize();

        // Prepare for the next kernel launch if needed
        remainingElements = gridSize;
        input = output;
    }
}

int main() {
    int sizes[] = {128, 512, 2048, 4096};

    for (int s = 0; s < 4; s++) {
        int n = sizes[s];
        float *h_input = (float*)malloc(n * sizeof(float));
        float *d_input, *d_output;

        // Initialize host input array
        for (int i = 0; i < n; i++) h_input[i] = 1.0f + (i % 5);

        // Allocate device memory
        cudaMalloc(&d_input, n * sizeof(float));
        cudaMalloc(&d_output, ((n + 511) / 512) * sizeof(float));

        // Copy data from host to device
        cudaMemcpy(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice);

        // Record start time
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        // Perform reduction using multi-kernel approach
        gpuReduction(d_input, d_output, n);

        // Record stop time
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        // Copy the final result back to host
        float result;
        cudaMemcpy(&result, d_output, sizeof(float), cudaMemcpyDeviceToHost);

        // Print the result and time taken
        printf("Array Size: %d, GPU Reduction Multi-Kernel Sum: %f, Time: %f ms\n", n, result, milliseconds);

        // Free memory
        free(h_input);
        cudaFree(d_input);
        cudaFree(d_output);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    return 0;
}
