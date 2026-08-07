// lets go 
#include<iostream>
#include<cstdlib>
#include<hip/hip_runtime.h>
using namespace std;

# define CHECK(cmd) \
{ \
    hipError_t err = cmd; \
    if (err != hipSuccess){ \
        std::cout << hipGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

#ifndef TILE_SIZE
#define TILE_SIZE 16
#endif

#ifndef M
#define M 4096
#endif

#ifndef N
#define N 4096
#endif

#ifndef K
#define K 4096
#endif

__global__ void matmul(int* A, int* B, int* C,
                                        int X, int Y, int Z)
{
    __shared__ int As[TILE_SIZE][TILE_SIZE];
    __shared__ int Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    int sum = 0;

    for (int tile = 0; tile < (Y + TILE_SIZE - 1) / TILE_SIZE; tile++)
    {
        int tiled_col_A = tile * TILE_SIZE + threadIdx.x;
        int tiled_row_B = tile * TILE_SIZE + threadIdx.y;
        // Load A tile
        if (row < X && tiled_col_A < Y)
            As[threadIdx.y][threadIdx.x] = A[row * Y + tiled_col_A];
        else
            As[threadIdx.y][threadIdx.x] = 0;

        // Load B tile
        if (tiled_row_B < Y && col < Z)
            Bs[threadIdx.y][threadIdx.x] = B[tiled_row_B * Z + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0;

        __syncthreads();
        #pragma unroll 16
        for (int k = 0; k < TILE_SIZE; k++)
        {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < X && col < Y)
        C[row * Z + col] = sum;
}

int main(){
    int* A = new int[M * K];
    int* B = new int[K * N];
    int* C = new int[M * N];
    int* C_cpu = new int[M * N];

    //random filling of A
    for (int row = 0; row < M; row++){
        for (int col = 0; col < K; col++){
            A[K * row + col] = rand() % 10;
        }
    }
    //random filling of B
    for (int row = 0; row < K; row++){
        for (int col = 0; col < N; col++){
            B[N * row + col] = rand() % 10;
        }
    }

    int *d_a, *d_b, *d_c;
    CHECK(hipMalloc(&d_a, M * K * sizeof(int)));
    CHECK(hipMalloc(&d_b, K * N * sizeof(int)));
    CHECK(hipMalloc(&d_c, M * N * sizeof(int)));

    //copying our stuff from cpu to device
    CHECK(hipMemcpy(d_a, A, M * K * sizeof(int), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(d_b, B, K * N * sizeof(int), hipMemcpyHostToDevice));
    
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    hipEventRecord(start);

    hipLaunchKernelGGL(matmul, blocks, threads, 0, 0,
                    d_a, d_b, d_c, M, K, N);

    hipEventRecord(stop);
    hipEventSynchronize(stop);

    float ms = 0;
    hipEventElapsedTime(&ms, start, stop);

    cout << "Kernel time: " << ms << " ms" << endl;    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());

    // Copy result back to host
    CHECK(hipMemcpy(C, d_c, M * N * sizeof(int), hipMemcpyDeviceToHost));

    CHECK(hipFree(d_a));
    CHECK(hipFree(d_b));
    CHECK(hipFree(d_c));
    delete[] A;
    delete[] B;
    delete[] C;
    delete[] C_cpu;

    return 0;
}