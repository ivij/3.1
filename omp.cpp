#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <cstdlib>
#include <time.h>
#include <chrono>

using namespace std::chrono;
using namespace std;

#define NUM_THREADS 10

int size = 1000;

void randomVector(int vector[], int size, char name)
{
    if(name == C)
    {
        for( int i = 0; i < size; ++i)
        {
            for( int j = 0;  j < size; ++j)
            {
                vector[i][j] = 0;
            }
        }
    }
    else
    {
        for( int i = 0; i < size; ++i)
        {
            for( int j = 0;  j < size; ++j)
            {
                vector[i][j] = rand() % 100;
            }
        }
    }
}

int main(int argc, char** argv) 
{
    int numtasks, rank, name_len, tag=1; 
    char name[MPI_MAX_PROCESSOR_NAME];
    char message[100];
    omp_set_num_threads(NUM_THREADS);

    // Initialize the MPI environment
    MPI_Init(&argc,&argv);

    // Get the number of tasks/process
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

    // Get the rank
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Find the processor name
    MPI_Get_processor_name(name, &name_len);

    strcpy (message, "Hello World!");


    int **A,**B,**C;

    auto start = high_resolution_clock::now();

    if(rank == 0)
    {
        A = (int **)  malloc(size * sizeof(int));
        B = (int **)  malloc(size * sizeof(int));
        C = (int **)  malloc(size * sizeof(int));

        for (int i = 0; i < size; i++)
        {
            A[i] = (int *) malloc(size * sizeof(int));
            B[i] = (int *) malloc(size * sizeof(int));
            C[i] = (int *) malloc(size * sizeof(int));
        }

        randomVector(A, size, A);
        randomVector(B, size, B);
        randomVector(C, size, C);

    }
    int partition_size = size/numtasks;
    int **sA,**sB,**sC;

    sA = (int **) malloc(partition_size * sizeof(int));
    sB = (int **) malloc(partition_size * sizeof(int));
    sC = (int **) malloc(partition_size * sizeof(int));

    for (int i = 0; i < partition_size; i++)
    {
        sA[i] = (int *) malloc(partition_size * sizeof(int));
        sB[i] = (int *) malloc(partition_size * sizeof(int));
        sC[i] = (int *) malloc(partition_size * sizeof(int));
    }

    MPI_Scatter(A, partition_size, MPI_INT, sA, partition_size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(B, partition_size, MPI_INT, sB, partition_size, MPI_INT, 0, MPI_COMM_WORLD);
    
    int threads;

    #pragma omp parallel default(none) firstprivate(size, sA, sB) shared (sC,threads)
    {
        int threads = omp_get_thread_num();
        printf("***** Thread id: %d *****\n", threads);

        #pragma omp for schedule(static)
        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < size; j++)
            {
                for (int k = 0; k < size; k++)
                {
                    sC[i][j] += sA[i][k] * sB[k][j];
                }
            }
        }
    }

    //MPI_Gather(sC, partition_size, MPI_INT, C, partition_size, MPI_INT, 0, MPI_COMM_WORLD);

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);

    MPI_Reduce(sC, C, partition_size, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        cout << "Time taken by function: "<< duration.count() << " microseconds "<< endl;
        int temp = 0;
        for (int i = 0; i < size; i++)
        {
            temp += C[i];
        }
        cout <<"sum "<< temp <<endl;
    }

    MPI_Finalize();
}