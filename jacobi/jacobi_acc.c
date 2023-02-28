#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#ifndef ITER_MAX
#define ITER_MAX 10000
#endif

#ifndef ERROR_MAX
#define ERROR_MAX 0.0001
#endif

#ifndef N
#define N 1250
#endif

#ifndef M
#define M N
#endif

#define UP_WALL 1.
#define LO_WALL 1.
#define LE_WALL 0.
#define RI_WALL 0.
#define CENTER 0.

void fill_grid(double **restrict grid)
{
    int i, j;

    for (j = 0; j < M; ++j) {
        grid[0][j] = UP_WALL;
        grid[N - 1][j] = LO_WALL;
    }

    for (i = 1; i < N - 1; ++i) {
        grid[i][0] = LE_WALL;
        grid[i][M - 1] = RI_WALL;
    }

    for (i = 1; i < N - 1; ++i)
        for (j = 1; j < M - 1; ++j)
            grid[i][j] = CENTER;
}

int main(void)
{
    double **restrict grid, **restrict new_grid, error = ERROR_MAX + 1.;

    int i, j, iter = 0;

    struct timeval start_time, end_time;

    long long elapsed_time;

    grid = malloc(N * sizeof(*grid));
    new_grid = malloc(N * sizeof(*new_grid));

    for (i = 0; i < N; ++i) {
        grid[i] = malloc(M * sizeof(**grid));
        new_grid[i] = malloc(M * sizeof(**new_grid));
    }

    fill_grid(grid);

    gettimeofday(&start_time, NULL);

#pragma acc data copy(grid[0:N][0:M]) create(new_grid[0:N][0:M])
    while (error > ERROR_MAX && iter < ITER_MAX) {
        error = 0.;

#pragma acc parallel loop reduction(max:error) //collapse(2)
        for (i = 1; i < N - 1; ++i) {
            for (j = 1; j < M - 1; ++j) {
                new_grid[i][j] = 0.25 * (grid[i - 1][j] + grid[i + 1][j]
                                       + grid[i][j - 1] + grid[i][j + 1]);

                error = fmax(error, fabs(new_grid[i][j] - grid[i][j]));
            }
        }

#pragma acc parallel loop //collapse(2)
        for (i = 1; i < N - 1; i++)
            for (j = 1; j < M - 1; j++)
                grid[i][j] = new_grid[i][j];

        ++iter;
    }

    gettimeofday(&end_time, NULL);

    elapsed_time = (end_time.tv_sec - start_time.tv_sec) * 1000000ll
                 + end_time.tv_usec - start_time.tv_usec;

    printf("%f\n", elapsed_time * 1e-6);

#ifdef PRINTR
#ifndef RSLT_FNAME
#define RSLT_FNAME "jacobi_acc.rslt"
#endif
#ifndef PRECISION
#define PRECISION 6
#endif
    FILE *stream = fopen(RSLT_FNAME, "w");
    for (i = 0; i < N; ++i) {
        for (j = 0; j < M; ++j)
            fprintf(stream, "%.*f ", PRECISION, grid[i][j]);
        fprintf(stream, "\n");
    }
    fclose(stream);
#endif

    for (i = 0; i < N; ++i) {
        free(grid[i]);
        free(new_grid[i]);
    }
    free(grid);
    free(new_grid);

    return 0;
}
