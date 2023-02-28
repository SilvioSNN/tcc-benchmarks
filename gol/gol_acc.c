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

#pragma acc routine seq
char get_new_state(char **restrict grid, int i, int j)
{
    char neighbors = grid[i - 1][j] + grid[i + 1][j]
                   + grid[i][j - 1] + grid[i][j + 1]
                   + grid[i - 1][j - 1] + grid[i - 1][j + 1]
                   + grid[i + 1][j - 1] + grid[i + 1][j + 1];

    if (grid[i][j] == 1 && (neighbors == 2 || neighbors == 3))
        return 1;
    else if (grid[i][j] == 0 && neighbors == 3)
        return 1;
    return 0;
}

int main(void)
{
    char **restrict grid, **restrict new_grid;

    int i, j, gen;

    struct timeval start_time, end_time;

    long long elapsed_time, population = 0;

    grid = malloc((GRID_SIZE + 2) * sizeof(*grid));
    new_grid = malloc((GRID_SIZE + 2) * sizeof(*new_grid));

    for (i = 0; i < GRID_SIZE + 2; ++i) {
        grid[i] = malloc((GRID_SIZE + 2) * sizeof(**grid));
        new_grid[i] = malloc((GRID_SIZE + 2) * sizeof(**new_grid));
    }

    srand(SEED);

    for (i = 1; i < GRID_SIZE + 1; ++i)
        for (j = 1; j < GRID_SIZE + 1; ++j)
            grid[i][j] = rand() % 2;

    gettimeofday(&start_time, NULL);

#pragma acc data copy(grid[0:GRID_SIZE+2][0:GRID_SIZE+2]) \
                 create(new_grid[0:GRID_SIZE+2][0:GRID_SIZE+2])
{
    for (gen = 0; gen < NUM_GENS; ++gen) {
#pragma acc parallel loop
        for (i = 1; i < GRID_SIZE + 1; ++i) {
            grid[i][0] = grid[i][GRID_SIZE];
            grid[i][GRID_SIZE + 1] = grid[i][1];
        }

#pragma acc parallel loop
        for (j = 0; j < GRID_SIZE + 2; ++j){
            grid[0][j] = grid[GRID_SIZE][j];
            grid[GRID_SIZE + 1][j] = grid[1][j];
        }

#pragma acc parallel loop //collapse(2)
        for (i = 1; i < GRID_SIZE + 1; ++i)
            for (j = 1; j < GRID_SIZE + 1; ++j)
                new_grid[i][j] = get_new_state(grid, i, j);

#pragma acc parallel loop //collapse(2)
        for (i = 1; i < GRID_SIZE + 1; ++i)
            for (j = 1; j < GRID_SIZE + 1; ++j)
                grid[i][j] = new_grid[i][j];
    }

#pragma acc parallel loop reduction(+:population)
    for (i = 1; i < GRID_SIZE + 1; ++i)
        for (j = 1; j < GRID_SIZE + 1; ++j)
            population += grid[i][j];
}

    gettimeofday(&end_time, NULL);

    elapsed_time = (end_time.tv_sec - start_time.tv_sec) * 1000000ll
                 + end_time.tv_usec - start_time.tv_usec;

    printf("%f\n", elapsed_time * 1e-6);

#ifdef PRINTR
#ifndef RESULT_FILE_NAME
#define RESULT_FILE_NAME "gol_acc.rslt"
#endif
    FILE *stream = fopen(RESULT_FILE_NAME, "w");
    fprintf(stream, "%lld\n", population);
    for (i = 1; i < GRID_SIZE + 1; ++i) {
        for (j = 1; j < GRID_SIZE + 1; ++j)
            fprintf(stream, "%d", grid[i][j]);
        fprintf(stream, "\n");
    }
    fclose(stream);
#endif

    for (i = 0; i < GRID_SIZE; ++i) {
        free(grid[i]);
        free(new_grid[i]);
    }
    free(grid);
    free(new_grid);

    return 0;
}
