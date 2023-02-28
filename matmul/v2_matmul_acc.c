#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#ifndef MAT_LEN
#define MAT_LEN 1250
#endif

#ifndef SEED
#define SEED 42
#endif

#define IND(i, j) ((i) * (MAT_LEN) + (j))

int main(void)
{
    float *restrict A, *restrict B, *restrict C;

    int i, j, k;

    struct timeval start_time, end_time;

    long long elapsed_time;

    srand(SEED);

    A = malloc(MAT_LEN * MAT_LEN * sizeof(*A));
    B = malloc(MAT_LEN * MAT_LEN * sizeof(*B));
    C = malloc(MAT_LEN * MAT_LEN * sizeof(*C));

    for (i = 0; i < MAT_LEN; ++i) {
        for (j = 0; j < MAT_LEN; ++j) {
            A[IND(i, j)] = (float) rand() / RAND_MAX * 10.f;
            B[IND(i, j)] = (float) rand() / RAND_MAX * 10.f;
            C[IND(i, j)] = 0.f;
        }
    }

    gettimeofday(&start_time, NULL);

#pragma acc parallel loop copyin(A[0:MAT_LEN*MAT_LEN], B[0:MAT_LEN*MAT_LEN]) \
                          copy(C[0:MAT_LEN*MAT_LEN]) collapse(2)
    for (i = 0; i < MAT_LEN; ++i)
        for (j = 0; j < MAT_LEN; ++j)
            for (k = 0; k < MAT_LEN; ++k)
                C[IND(i, j)] += A[IND(i, k)] * B[IND(k, j)];

    gettimeofday(&end_time, NULL);

    elapsed_time = (end_time.tv_sec - start_time.tv_sec) * 1000000ll
                 + end_time.tv_usec - start_time.tv_usec;

    printf("%f\n", elapsed_time * 1e-6);

#ifdef PRINTR
#ifndef RESULT_FILE_NAME
#define RESULT_FILE_NAME "v2_matmul_acc.rslt"
#endif
#ifndef PRECISION
#define PRECISION 6
#endif
    FILE *stream = fopen(RESULT_FILE_NAME, "w");
    for (i = 0; i < MAT_LEN; ++i) {
        for (j = 0; j < MAT_LEN; ++j)
            fprintf(stream, "%.*f ", PRECISION, C[IND(i, j)]);
        fprintf(stream, "\n");
    }
    fclose(stream);
#endif

    free(A);
    free(B);
    free(C);

    return 0;
}
