#include <stdio.h>
#include <cuda.h>

__global__ void gpuHistogramStrided(int *input, int *hist, int n, int bins) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = tid; i < n; i += stride) {
        atomicAdd(&hist[input[i] % bins], 1);
    }
}

int main() {
    int n = 1024;
    int bins = 10;
    int *h_input = (int*)malloc(n * sizeof(int));
    int *d_input, *d_hist;
    for (int i = 0; i < n; i++) h_input[i] = i % bins;

    cudaMalloc(&d_input, n * sizeof(int));
    cudaMalloc(&d_hist, bins * sizeof(int));
    cudaMemcpy(d_input, h_input, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_hist, 0, bins * sizeof(int));

    dim3 blockSize(256);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x);

    gpuHistogramStrided<<<gridSize, blockSize>>>(d_input, d_hist, n, bins);
    cudaDeviceSynchronize();

    int *h_hist = (int*)malloc(bins * sizeof(int));
    cudaMemcpy(h_hist, d_hist, bins * sizeof(int), cudaMemcpyDeviceToHost);

    printf("GPU Histogram Strided:\n");
    for (int i = 0; i < bins; i++) printf("Bin %d: %d\n", i, h_hist[i]);

    cudaFree(d_input);
    cudaFree(d_hist);
    free(h_input);
    free(h_hist);
    return 0;
}
