#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#ifndef GRID_SIZE
#define GRID_SIZE 1250
#endif

#ifndef NUM_GENS
#define NUM_GENS 2000
#endif

#ifndef SEED
#define SEED 42
#endif

#define IND(i, j) ((i) * (GRID_SIZE + 2) + (j))

char get_new_state(char *restrict grid, int i, int j)
{
    char neighbors = grid[IND(i - 1, j)] + grid[IND(i + 1, j)]
                   + grid[IND(i, j - 1)] + grid[IND(i, j + 1)]
                   + grid[IND(i - 1, j - 1)] + grid[IND(i - 1, j + 1)]
                   + grid[IND(i + 1, j - 1)] + grid[IND(i + 1, j + 1)];

    if (grid[IND(i, j)] == 1 && (neighbors == 2 || neighbors == 3))
        return 1;
    else if (grid[IND(i, j)] == 0 && neighbors == 3)
        return 1;
    return 0;
}
#pragma omp declare target(get_new_state)

int main(void)
{
    char *restrict grid, *restrict new_grid;

    int i, j, gen;

    struct timeval start_time, end_time;

    long long elapsed_time, population = 0;

    grid = malloc((GRID_SIZE + 2) * (GRID_SIZE + 2) * sizeof(*grid));
    new_grid = malloc((GRID_SIZE + 2) * (GRID_SIZE + 2) * sizeof(*new_grid));

    srand(SEED);

    for (i = 1; i < GRID_SIZE + 1; ++i)
        for (j = 1; j < GRID_SIZE + 1; ++j)
            grid[IND(i, j)] = rand() % 2;

    gettimeofday(&start_time, NULL);

#pragma omp target data map(tofrom:grid[0:(GRID_SIZE+2)*(GRID_SIZE+2)]) \
                        map(alloc:new_grid[0:(GRID_SIZE+2)*(GRID_SIZE+2)])
{
    for (gen = 0; gen < NUM_GENS; ++gen) {
#pragma omp target teams distribute parallel for
        for (i = 1; i < GRID_SIZE + 1; ++i) {
            grid[IND(i, 0)] = grid[IND(i, GRID_SIZE)];
            grid[IND(i, GRID_SIZE + 1)] = grid[IND(i, 1)];
        }

#pragma omp target teams distribute parallel for
        for (j = 0; j < GRID_SIZE + 2; ++j){
            grid[IND(0, j)] = grid[IND(GRID_SIZE, j)];
            grid[IND(GRID_SIZE + 1, j)] = grid[IND(1, j)];
        }

#pragma omp target teams distribute parallel for collapse(2)
        for (i = 1; i < GRID_SIZE + 1; ++i)
            for (j = 1; j < GRID_SIZE + 1; ++j)
                new_grid[IND(i, j)] = get_new_state(grid, i, j);

#pragma omp target teams distribute parallel for collapse(2)
        for (i = 1; i < GRID_SIZE + 1; ++i)
            for (j = 1; j < GRID_SIZE + 1; ++j)
                grid[IND(i, j)] = new_grid[IND(i, j)];
    }

#pragma omp target teams distribute parallel for reduction(+:population)
    for (i = 1; i < GRID_SIZE + 1; ++i)
        for (j = 1; j < GRID_SIZE + 1; ++j)
            population += grid[IND(i, j)];
}

    gettimeofday(&end_time, NULL);

    elapsed_time = (end_time.tv_sec - start_time.tv_sec) * 1000000ll
                 + end_time.tv_usec - start_time.tv_usec;

    printf("%f\n", elapsed_time * 1e-6);

#ifdef PRINTR
#ifndef RESULT_FILE_NAME
#define RESULT_FILE_NAME "v2_gol_omp.rslt"
#endif
    FILE *stream = fopen(RESULT_FILE_NAME, "w");
    fprintf(stream, "%lld\n", population);
    for (i = 1; i < GRID_SIZE + 1; ++i) {
        for (j = 1; j < GRID_SIZE + 1; ++j)
            fprintf(stream, "%d", grid[IND(i, j)]);
        fprintf(stream, "\n");
    }
    fclose(stream);
#endif

    free(grid);
    free(new_grid);

    return 0;
}
