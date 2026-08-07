#include<iostream>
#include<hip/hip_runtime.h>
#include<cstdlib>

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
#define TILE_SIZE 8
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
    constexpr int VEC = 4;

    __shared__ int As[TILE_SIZE][TILE_SIZE];

    // B is stored transposed:
    // Bs[output_column][k]
    __shared__ int Bs[TILE_SIZE][TILE_SIZE];

    const int tx = threadIdx.x;  // 0 ... TILE_SIZE/4 - 1
    const int ty = threadIdx.y;  // 0 ... TILE_SIZE - 1

    const int global_row =
        blockIdx.y * TILE_SIZE + ty;

    const int output_start_col =
        blockIdx.x * TILE_SIZE + tx * VEC;

    // Each thread computes four horizontal C elements.
    int sum[VEC] = {0, 0, 0, 0};

    const int number_of_tiles =
        (Y + TILE_SIZE - 1) / TILE_SIZE;

    for (int tile = 0; tile < number_of_tiles; tile++) {

        const int a_start_col =
            tile * TILE_SIZE + tx * VEC;

        if (global_row < X && a_start_col + 3 < Y) {

            const int4 a =
                *reinterpret_cast<const int4*>(
                    &A[global_row * Y + a_start_col]);

            As[ty][tx * VEC + 0] = a.x;
            As[ty][tx * VEC + 1] = a.y;
            As[ty][tx * VEC + 2] = a.z;
            As[ty][tx * VEC + 3] = a.w;

        } else {
            // Handles incomplete boundary tiles safely.
            for (int j = 0; j < VEC; j++) {
                const int a_col = a_start_col + j;

                As[ty][tx * VEC + j] =
                    (global_row < X && a_col < Y)
                    ? A[global_row * Y + a_col]
                    : 0;
            }
        }

        /*
         * Load B horizontally from global memory.
         * Then transpose while storing into shared memory.
         */
        const int b_global_row =
            tile * TILE_SIZE + ty;

        const int b_start_col =
            blockIdx.x * TILE_SIZE + tx * VEC;

        if (b_global_row < Y && b_start_col + 3 < Z) {

            const int4 b =
                *reinterpret_cast<const int4*>(
                    &B[b_global_row * N + b_start_col]);

            Bs[tx * VEC + 0][ty] = b.x;
            Bs[tx * VEC + 1][ty] = b.y;
            Bs[tx * VEC + 2][ty] = b.z;
            Bs[tx * VEC + 3][ty] = b.w;

        } else {
            for (int j = 0; j < VEC; j++) {
                const int b_col = b_start_col + j;

                Bs[tx * VEC + j][ty] =
                    (b_global_row < Y && b_col < Z)
                    ? B[b_global_row * Z + b_col]
                    : 0;
            }
        }

        __syncthreads();

        // Compute four horizontal output values.
        for (int k = 0; k < TILE_SIZE; k++) {

            const int a = As[ty][k];

            for (int j = 0; j < VEC; j++) {
                sum[j] +=
                    a * Bs[tx * VEC + j][k];
            }
        }

        __syncthreads();
    }

    // Store four horizontal values of C.
    if (global_row < X) {
        for (int j = 0; j < VEC; j++) {

            const int global_col =
                output_start_col + j;

            if (global_col < Z) {
                C[global_row * Z + global_col] =
                    sum[j];
            }
        }
    }
}

int main(){

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

    CHECK(hipMemcpy(d_a, A, bytes_A, hipMemcpyHostToDevice));

    CHECK(hipMemcpy(d_b, B, bytes_B, hipMemcpyHostToDevice));

    const dim3 threads(TILE_SIZE / 4, TILE_SIZE);
    const dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE,(M + TILE_SIZE - 1) / TILE_SIZE);

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
    CHECK(hipEventElapsedTime(&milliseconds, start, stop));

    std::cout << "Kernel time: " << milliseconds << " ms\n";

    CHECK(hipMemcpy(C, d_c, bytes_C, hipMemcpyDeviceToHost));

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