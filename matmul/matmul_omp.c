#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#ifndef MAT_LEN
#define MAT_LEN 1250
#endif

#ifndef SEED
#define SEED 42
#endif

int main(void)
{
    float **restrict A, **restrict B, **restrict C;

    int i, j, k;

    struct timeval start_time, end_time;

    long long elapsed_time;

    srand(SEED);

    A = malloc((MAT_LEN) * sizeof(*A));
    B = malloc((MAT_LEN) * sizeof(*B));
    C = malloc((MAT_LEN) * sizeof(*C));

    for (i = 0; i < MAT_LEN; ++i) {
        A[i] = malloc((MAT_LEN) * sizeof(**A));
        B[i] = malloc((MAT_LEN) * sizeof(**B));
        C[i] = malloc((MAT_LEN) * sizeof(**C));
    }

    for (i = 0; i < MAT_LEN; ++i) {
        for (j = 0; j < MAT_LEN; ++j) {
            A[i][j] = (float) rand() / RAND_MAX * 10.f;
            B[i][j] = (float) rand() / RAND_MAX * 10.f;
            C[i][j] = 0.f;
        }
    }

    gettimeofday(&start_time, NULL);

#pragma omp target teams distribute parallel for \
                   map(to:A[0:MAT_LEN][0:MAT_LEN], B[0:MAT_LEN][0:MAT_LEN]) \
                   map(tofrom:C[0:MAT_LEN][0:MAT_LEN]) collapse(2)
    for (i = 0; i < MAT_LEN; ++i)
        for (j = 0; j < MAT_LEN; ++j)
            for (k = 0; k < MAT_LEN; ++k)
                C[i][j] += A[i][k] * B[k][j];

    gettimeofday(&end_time, NULL);

    elapsed_time = (end_time.tv_sec - start_time.tv_sec) * 1000000ll
                 + end_time.tv_usec - start_time.tv_usec;

    printf("%f\n", elapsed_time * 1e-6);

#ifdef PRINTR
#ifndef RESULT_FILE_NAME
#define RESULT_FILE_NAME "matmul_omp.rslt"
#endif
#ifndef PRECISION
#define PRECISION 6
#endif
    FILE *stream = fopen(RESULT_FILE_NAME, "w");
    for (i = 0; i < MAT_LEN; ++i) {
        for (j = 0; j < MAT_LEN; ++j)
            fprintf(stream, "%.*f ", PRECISION, C[i][j]);
        fprintf(stream, "\n");
    }
    fclose(stream);
#endif

    for (i = 0; i < MAT_LEN; ++i) {
        free(A[i]);
        free(B[i]);
        free(C[i]);
    }
    free(A);
    free(B);
    free(C);

    return 0;
}
