#include <stdio.h>
#include <cuda.h>

__global__ void gpuHistogramNonStrided(int *input, int *hist, int n, int bins) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid < n) {
        atomicAdd(&hist[input[tid] % bins], 1);
    }
}

int main() {
    int sizes[] = {128, 512, 2048, 4096};
    int bins = 10;

    for (int s = 0; s < 4; s++) {
        int n = sizes[s];
        int *h_input = (int*)malloc(n * sizeof(int));
        int *h_hist = (int*)calloc(bins, sizeof(int));  // Initialize histogram on host

        // Initialize input array on host
        for (int i = 0; i < n; i++) h_input[i] = i % bins;

        int *d_input, *d_hist;
        cudaMalloc(&d_input, n * sizeof(int));
        cudaMalloc(&d_hist, bins * sizeof(int));

        // Copy input data from host to device
        cudaMemcpy(d_input, h_input, n * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemset(d_hist, 0, bins * sizeof(int));  // Initialize histogram on device

        dim3 blockSize(256);
        dim3 gridSize((n + blockSize.x - 1) / blockSize.x);

        // Record start time
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        // Launch kernel for non-strided GPU histogram
        gpuHistogramNonStrided<<<gridSize, blockSize>>>(d_input, d_hist, n, bins);
        cudaDeviceSynchronize();

        // Record stop time
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        // Copy histogram result back to host
        cudaMemcpy(h_hist, d_hist, bins * sizeof(int), cudaMemcpyDeviceToHost);

        // Print the result and time taken
        printf("Array Size: %d, Time taken (GPU Histogram Non-Strided): %f ms\n", n, milliseconds);
        printf("Histogram:\n");
        for (int i = 0; i < bins; i++) {
            printf("Bin %d: %d\n", i, h_hist[i]);
        }
        printf("\n");

        // Free memory
        free(h_input);
        free(h_hist);
        cudaFree(d_input);
        cudaFree(d_hist);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    return 0;
}
