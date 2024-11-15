#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>

void cpuReduction(float *input, float *sum, float *product, float *min, float *max, int n) {
    *sum = 0;
    *product = 1;
    *min = FLT_MAX;
    *max = FLT_MIN;

    for (int i = 0; i < n; i++) {
        *sum += input[i];
        *product *= input[i];
        if (input[i] < *min) *min = input[i];
        if (input[i] > *max) *max = input[i];
    }
}

int main() {
    int sizes[] = {128, 512, 2048, 4096};

    for (int s = 0; s < 4; s++) {
        int n = sizes[s];
        float *input = (float*)malloc(n * sizeof(float));

        for (int i = 0; i < n; i++) {
            input[i] = 1.0f + (i % 5);
        }

        float sum, product, min, max;

        clock_t start = clock();

        cpuReduction(input, &sum, &product, &min, &max, n);

        clock_t end = clock();
        double total_time = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;

        printf("Array Size: %d\n", n);
        printf("Sum: %f, Product: %f, Min: %f, Max: %f\n", sum, product, min, max);
        printf("Time taken (CPU Reduction): %f ms\n\n", total_time);

        free(input);
    }

    return 0;
}
