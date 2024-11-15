#include <stdio.h>
#include <cuda.h>

__global__ void gpuReductionLessDivergence(float *input, float *output, int n) {
    extern __shared__ float sharedData[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    sharedData[tid] = (i < n) ? input[i] + input[i + blockDim.x] : 0;
    __syncthreads();

    for (int s = 1; s < blockDim.x; s *= 2) {
        int index = 2 * s * tid;
        if (index < blockDim.x) {
            sharedData[index] += sharedData[index + s];
        }
        __syncthreads();
    }

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

        gpuReductionLessDivergence<<<gridSize, blockSize, sharedMemSize>>>(input, output, remainingElements);
        cudaDeviceSynchronize();

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

        for (int i = 0; i < n; i++) h_input[i] = 1.0f + (i % 5);

        cudaMalloc(&d_input, n * sizeof(float));
        cudaMalloc(&d_output, ((n + 511) / 512) * sizeof(float));

        cudaMemcpy(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        gpuReduction(d_input, d_output, n);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        float result;
        cudaMemcpy(&result, d_output, sizeof(float), cudaMemcpyDeviceToHost);

        printf("Array Size: %d, GPU Reduction with Less Thread Divergence Sum: %f, Time: %f ms\n", n, result, milliseconds);

        free(h_input);
        cudaFree(d_input);
        cudaFree(d_output);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    return 0;
}
