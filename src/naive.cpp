# include <iostream>
# include <cstdlib>
# include <iostream>
# include <chrono>

using namespace std;

#ifndef M 
#define M 1024
#endif

#ifndef N
#define N 1024
#endif 

#ifndef K
#define K 1024
#endif

void naive_matmul(int* A, int* B, int* C, int rows, int inner, int columns){
    for (int m = 0; m < rows; m++){
        for (int n = 0; n < columns; n++){
            int sum = 0;
            for (int k = 0; k < inner; k++){
                sum += A[m * inner + k] * B[k * columns + n];
            }
            C[m * columns + n] = sum;
        }
    }
}

int main()
{
    int* A = new int[static_cast<size_t>(M) * K];
    int* B = new int[static_cast<size_t>(K) * N];
    int* C = new int[static_cast<size_t>(M) * N];

    for (size_t i = 0; i < static_cast<size_t>(M) * K; i++) {
        A[i] = std::rand() % 10;
    }

    for (size_t i = 0; i < static_cast<size_t>(K) * N; i++) {
        B[i] = std::rand() % 10;
    }

    auto start = std::chrono::high_resolution_clock::now();

    naive_matmul(A, B, C, M, K, N);

    long long checksum = 0;
    for (size_t i = 0; i < static_cast<size_t>(M) * N; i++) {
        checksum += C[i];
    }

    auto stop = std::chrono::high_resolution_clock::now();

    double milliseconds = std::chrono::duration<double, std::milli>(stop - start).count();

    std::cout << "Matrix Size : " << M << " x " << K << " x " << N << '\n';

    std::cout << "Execution Time : " << milliseconds << " ms\n";

    std::cout << "Checksum : " << checksum << '\n';

    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}