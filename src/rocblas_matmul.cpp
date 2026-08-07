#include <cstdlib>
#include <iostream>
#include <vector>

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#define CHECK_HIP(command)                                                   \
    do {                                                                     \
        hipError_t error = (command);                                        \
        if (error != hipSuccess) {                                           \
            std::cerr << "HIP error: " << hipGetErrorString(error)           \
                      << " at " << __FILE__ << ":" << __LINE__ << '\n';       \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

#define CHECK_ROCBLAS(command)                                                \
    do {                                                                     \
        rocblas_status status = (command);                                   \
        if (status != rocblas_status_success) {                              \
            std::cerr << "rocBLAS error: " << status                         \
                      << " at " << __FILE__ << ":" << __LINE__ << '\n';       \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

int main()
{
    // Use the same dimensions as your other implementations.
    constexpr int M = 4096;
    constexpr int K = 4096;
    constexpr int N = 4096;

    constexpr int warmup_runs = 5;
    constexpr int measured_runs = 20;

    const std::size_t size_A =
        static_cast<std::size_t>(M) * K * sizeof(float);

    const std::size_t size_B =
        static_cast<std::size_t>(K) * N * sizeof(float);

    const std::size_t size_C =
        static_cast<std::size_t>(M) * N * sizeof(float);

    std::vector<float> h_A(static_cast<std::size_t>(M) * K);
    std::vector<float> h_B(static_cast<std::size_t>(K) * N);
    std::vector<float> h_C(static_cast<std::size_t>(M) * N);

    // Initialize the matrices.
    std::srand(0);

    for (float& value : h_A) {
        value = static_cast<float>(std::rand() % 10);
    }

    for (float& value : h_B) {
        value = static_cast<float>(std::rand() % 10);
    }

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;

    CHECK_HIP(hipMalloc(&d_A, size_A));
    CHECK_HIP(hipMalloc(&d_B, size_B));
    CHECK_HIP(hipMalloc(&d_C, size_C));

    CHECK_HIP(
        hipMemcpy(d_A, h_A.data(), size_A, hipMemcpyHostToDevice));

    CHECK_HIP(
        hipMemcpy(d_B, h_B.data(), size_B, hipMemcpyHostToDevice));

    CHECK_HIP(hipMemset(d_C, 0, size_C));

    rocblas_handle handle = nullptr;
    CHECK_ROCBLAS(rocblas_create_handle(&handle));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    /*
       Your matrices use row-major storage:

           C = A × B

       rocBLAS uses column-major storage by default.

       Therefore, we reverse A and B and compute:

           C^T = B^T × A^T

       The resulting memory layout is the same as row-major C.
    */

    auto run_rocblas_gemm = [&]() {
        CHECK_ROCBLAS(
            rocblas_sgemm(
                handle,
                rocblas_operation_none,
                rocblas_operation_none,

                // rocBLAS sees C^T as an N x M matrix.
                N,
                M,
                K,

                &alpha,

                // B is passed first.
                d_B,
                N,

                // A is passed second.
                d_A,
                K,

                &beta,
                d_C,
                N));
    };

    // Warm-up runs are not timed.
    for (int run = 0; run < warmup_runs; ++run) {
        run_rocblas_gemm();
    }

    CHECK_HIP(hipDeviceSynchronize());

    hipEvent_t start;
    hipEvent_t stop;

    CHECK_HIP(hipEventCreate(&start));
    CHECK_HIP(hipEventCreate(&stop));

    CHECK_HIP(hipEventRecord(start));

    for (int run = 0; run < measured_runs; ++run) {
        run_rocblas_gemm();
    }

    CHECK_HIP(hipEventRecord(stop));
    CHECK_HIP(hipEventSynchronize(stop));

    float total_time_ms = 0.0f;

    CHECK_HIP(
        hipEventElapsedTime(&total_time_ms, start, stop));

    const float average_time_ms =
        total_time_ms / static_cast<float>(measured_runs);

    std::cout << "Kernel time: "
              << average_time_ms
              << " ms\n";

    // Optional: calculate achieved performance.
    const double operations =
        2.0 * static_cast<double>(M) *
        static_cast<double>(N) *
        static_cast<double>(K);

    const double seconds =
        static_cast<double>(average_time_ms) / 1000.0;

    const double tera_flops =
        operations / seconds / 1.0e12;

    std::cout << "Performance: "
              << tera_flops
              << " TFLOP/s\n";

    // Copying C back is outside the measured region.
    CHECK_HIP(
        hipMemcpy(h_C.data(), d_C, size_C, hipMemcpyDeviceToHost));

    CHECK_HIP(hipEventDestroy(start));
    CHECK_HIP(hipEventDestroy(stop));

    CHECK_ROCBLAS(rocblas_destroy_handle(handle));

    CHECK_HIP(hipFree(d_A));
    CHECK_HIP(hipFree(d_B));
    CHECK_HIP(hipFree(d_C));

    return 0;
}