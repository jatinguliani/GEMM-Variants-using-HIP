// lets go
#include <iostream>
#include <cstdlib>
#include <hip/hip_runtime.h>
using namespace std;

#define CHECK(cmd)                                      \
{                                                       \
    hipError_t err = cmd;                               \
    if (err != hipSuccess) {                            \
        std::cout << hipGetErrorString(err) << endl;    \
        exit(EXIT_FAILURE);                             \
    }                                                   \
}

#ifndef TILE_SIZE
#define TILE_SIZE 16
#endif

#ifndef REG_M
#define REG_M 2
#endif

__global__ void matmul(int* A, int* B, int* C,
                       int M, int K, int N)
{
    __shared__ int As[TILE_SIZE][TILE_SIZE];
    __shared__ int Bs[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int col = blockIdx.x * TILE_SIZE + tx;
    int row_base = blockIdx.y * TILE_SIZE + ty;

    int sum[REG_M] = {0};

    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; tile++)
    {
        int tiled_col_A = tile * TILE_SIZE + tx;

        // Load A: each thread loads REG_M vertical rows
        for (int i = 0; i < REG_M; i++) {
            int local_a_row = ty + i * (TILE_SIZE / REG_M);
            int global_a_row = blockIdx.y * TILE_SIZE + local_a_row;

            if (global_a_row < M && tiled_col_A < K)
                As[local_a_row][tx] = A[global_a_row * K + tiled_col_A];
            else
                As[local_a_row][tx] = 0;
        }

        // Load B: each thread loads REG_M vertical rows
        for (int i = 0; i < REG_M; i++) {
            int local_b_row = ty + i * (TILE_SIZE / REG_M);
            int global_b_row = tile * TILE_SIZE + local_b_row;

            if (global_b_row < K && col < N)
                Bs[local_b_row][tx] = B[global_b_row * N + col];
            else
                Bs[local_b_row][tx] = 0;
        }

        __syncthreads();

        // Compute REG_M vertical C values
        for (int k = 0; k < TILE_SIZE; k++) {
            int b = Bs[k][tx];
            for (int i = 0; i < REG_M; i++) {
                int local_a_row = ty + i * (TILE_SIZE / REG_M);
                sum[i] += As[local_a_row][k] * b;
            }
        }

        __syncthreads();
    }

    // Store REG_M vertical C values
    for (int i = 0; i < REG_M; i++) {
        int local_c_row = ty + i * (TILE_SIZE / REG_M);
        int global_c_row = blockIdx.y * TILE_SIZE + local_c_row;

        if (global_c_row < M && col < N)
            C[global_c_row * N + col] = sum[i];
    }
}

void naive_matmul(int* A, int* B, int* C_cpu, int M, int K, int N)
{
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            int sum = 0;
            for (int k = 0; k < K; k++) {
                sum += A[m * K + k] * B[k * N + n];
            }
            C_cpu[m * N + n] = sum;
        }
    }
}

int main()
{
    int M = 4096;
    int N = 4096;
    int K = 4096;

    int* A = new int[M * K];
    int* B = new int[K * N];
    int* C = new int[M * N];
    int* C_cpu = new int[M * N];

    for (int row = 0; row < M; row++) {
        for (int col = 0; col < K; col++) {
            A[row * K + col] = rand() % 10;
        }
    }

    for (int row = 0; row < K; row++) {
        for (int col = 0; col < N; col++) {
            B[row * N + col] = rand() % 10;
        }
    }

    // naive_matmul(A, B, C_cpu, M, K, N);

    int *d_a, *d_b, *d_c;

    CHECK(hipMalloc(&d_a, M * K * sizeof(int)));
    CHECK(hipMalloc(&d_b, K * N * sizeof(int)));
    CHECK(hipMalloc(&d_c, M * N * sizeof(int)));

    CHECK(hipMemcpy(d_a, A, M * K * sizeof(int), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(d_b, B, K * N * sizeof(int), hipMemcpyHostToDevice));

    dim3 threads(TILE_SIZE, TILE_SIZE / REG_M);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE,
                (M + TILE_SIZE - 1) / TILE_SIZE);

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

    cout << "Kernel time: " << ms << " ms" << endl;
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());

    CHECK(hipMemcpy(C, d_c, M * N * sizeof(int), hipMemcpyDeviceToHost));

    // bool ok = true;

    // for (int i = 0; i < M * N; i++) {
    //     if (C[i] != C_cpu[i]) {
    //         cout << "Mismatch at " << i
    //              << " GPU=" << C[i]
    //              << " CPU=" << C_cpu[i] << endl;
    //         ok = false;
    //         break;
    //     }
    // }

    // if (ok)
    //     cout << "Correct!" << endl;

    // cout << "done" << endl;

    CHECK(hipFree(d_a));
    CHECK(hipFree(d_b));
    CHECK(hipFree(d_c));

    delete[] A;
    delete[] B;
    delete[] C;
    delete[] C_cpu;

    return 0;
}