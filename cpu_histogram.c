#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void cpuHistogram(int *input, int *hist, int n, int bins) {
    for (int i = 0; i < bins; i++) hist[i] = 0;
    for (int i = 0; i < n; i++) {
        hist[input[i] % bins]++;
    }
}

int main() {
    int sizes[] = {128, 512, 2048, 4096}; 
    int bins = 10;

    for (int s = 0; s < 4; s++) {
        int n = sizes[s];
        int *input = (int*)malloc(n * sizeof(int));
        int *hist = (int*)calloc(bins, sizeof(int));

        for (int i = 0; i < n; i++) input[i] = i % bins;

        clock_t start = clock();

        cpuHistogram(input, hist, n, bins);

        clock_t end = clock();
        double total_time = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;

        printf("Array Size: %d, Time taken (CPU Histogram): %f ms\n", n, total_time);
        printf("Histogram:\n");
        for (int i = 0; i < bins; i++) {
            printf("Bin %d: %d\n", i, hist[i]);
        }
        printf("\n");

        free(input);
        free(hist);
    }

    return 0;
}
