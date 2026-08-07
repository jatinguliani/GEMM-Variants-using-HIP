#include <cstdlib>
#include <hip/hip_runtime.h>
#include <iostream>

#define CHECK(cmd)                                                          \
    do {                                                                    \
        hipError_t err = (cmd);                                             \
        if (err != hipSuccess) {                                            \
            std::cerr << "HIP error: " << hipGetErrorString(err)            \
                      << " at " << __FILE__ << ":" << __LINE__ << '\n';      \
            std::exit(EXIT_FAILURE);                                        \
        }                                                                   \
    } while (0)

#ifndef TILE_SIZE
#define TILE_SIZE 16
#endif

#ifndef REG_M
#define REG_M 2
#endif

#ifndef REG_N
#define REG_N 2
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

__global__ void matmul(const int* A,
                       const int* B,
                       int* C,
                       int X,
                       int Y,
                       int Z)
{
    __shared__ int As[TILE_SIZE][TILE_SIZE];
    __shared__ int Bs[TILE_SIZE][TILE_SIZE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    int sum[REG_M][REG_N] = {};

    const int number_of_tiles =
        (Y + TILE_SIZE - 1) / TILE_SIZE;

    for (int tile = 0; tile < number_of_tiles; tile++) {

        // Load one TILE_SIZE × TILE_SIZE tile of A.
        for (int j = 0; j < REG_N; j++) {
            const int local_col = tx + j * (TILE_SIZE / REG_N);

            const int global_col = tile * TILE_SIZE + local_col;

            for (int i = 0; i < REG_M; i++) {
                const int local_row = ty + i * (TILE_SIZE / REG_M);
                const int global_row = blockIdx.y * TILE_SIZE + local_row;
                if (global_row < X && global_col < Y) {
                    As[local_row][local_col] = A[global_row * K + global_col];
                } else {
                    As[local_row][local_col] = 0;
                }
            }
        }

        // Load one TILE_SIZE × TILE_SIZE tile of B.
        for (int i = 0; i < REG_M; i++) {
            const int local_row =
                ty + i * (TILE_SIZE / REG_M);

            const int global_row =
                tile * TILE_SIZE + local_row;

            for (int j = 0; j < REG_N; j++) {
                const int local_col =
                    tx + j * (TILE_SIZE / REG_N);

                const int global_col =
                    blockIdx.x * TILE_SIZE + local_col;

                if (global_row < Y && global_col < Z) {
                    Bs[local_row][local_col] =
                        B[global_row * N + global_col];
                } else {
                    Bs[local_row][local_col] = 0;
                }
            }
        }

        __syncthreads();

        // Each thread computes REG_M × REG_N values of C.
        for (int k = 0; k < TILE_SIZE; k++) {
            for (int j = 0; j < REG_N; j++) {
                const int local_col =
                    tx + j * (TILE_SIZE / REG_N);

                const int b = Bs[k][local_col];

                for (int i = 0; i < REG_M; i++) {
                    const int local_row =
                        ty + i * (TILE_SIZE / REG_M);

                    sum[i][j] += As[local_row][k] * b;
                }
            }
        }

        __syncthreads();
    }

    // Store REG_M × REG_N output values per thread.
    for (int j = 0; j < REG_N; j++) {
        const int local_col =
            tx + j * (TILE_SIZE / REG_N);

        const int global_col =
            blockIdx.x * TILE_SIZE + local_col;

        for (int i = 0; i < REG_M; i++) {
            const int local_row =
                ty + i * (TILE_SIZE / REG_M);

            const int global_row =
                blockIdx.y * TILE_SIZE + local_row;

            if (global_row < X && global_col < Z) {
                C[global_row * Z + global_col] = sum[i][j];
            }
        }
    }
}

int main()
{

    const size_t bytes_A =
        static_cast<size_t>(M) * K * sizeof(int);

    const size_t bytes_B =
        static_cast<size_t>(K) * N * sizeof(int);

    const size_t bytes_C =
        static_cast<size_t>(M) * N * sizeof(int);

    int* A = new int[static_cast<size_t>(M) * K];
    int* B = new int[static_cast<size_t>(K) * N];
    int* C = new int[static_cast<size_t>(M) * N];

    for (size_t i = 0; i < static_cast<size_t>(M) * K; i++) {
        A[i] = std::rand() % 10;
    }

    for (size_t i = 0; i < static_cast<size_t>(K) * N; i++) {
        B[i] = std::rand() % 10;
    }

    int* d_a = nullptr;
    int* d_b = nullptr;
    int* d_c = nullptr;

    CHECK(hipMalloc(&d_a, bytes_A));
    CHECK(hipMalloc(&d_b, bytes_B));
    CHECK(hipMalloc(&d_c, bytes_C));

    CHECK(hipMemcpy(
        d_a, A, bytes_A, hipMemcpyHostToDevice));

    CHECK(hipMemcpy(
        d_b, B, bytes_B, hipMemcpyHostToDevice));

    const dim3 threads(
        TILE_SIZE / REG_N,
        TILE_SIZE / REG_M);

    const dim3 blocks(
        (N + TILE_SIZE - 1) / TILE_SIZE,
        (M + TILE_SIZE - 1) / TILE_SIZE);

    hipEvent_t start;
    hipEvent_t stop;

    CHECK(hipEventCreate(&start));
    CHECK(hipEventCreate(&stop));

    // Optional warm-up launch.
    hipLaunchKernelGGL(
        matmul,
        blocks,
        threads,
        0,
        0,
        d_a,
        d_b,
        d_c,
        M,
        K,
        N);

    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());

    CHECK(hipEventRecord(start));

    hipLaunchKernelGGL(
        matmul,
        blocks,
        threads,
        0,
        0,
        d_a,
        d_b,
        d_c,
        M,
        K,
        N);

    CHECK(hipGetLastError());
    CHECK(hipEventRecord(stop));
    CHECK(hipEventSynchronize(stop));

    float milliseconds = 0.0f;
    CHECK(hipEventElapsedTime(
        &milliseconds, start, stop));

    std::cout << "Kernel time: "
              << milliseconds << " ms\n";

    CHECK(hipMemcpy(
        C, d_c, bytes_C, hipMemcpyDeviceToHost));

    CHECK(hipEventDestroy(start));
    CHECK(hipEventDestroy(stop));

    CHECK(hipFree(d_a));
    CHECK(hipFree(d_b));
    CHECK(hipFree(d_c));

    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}