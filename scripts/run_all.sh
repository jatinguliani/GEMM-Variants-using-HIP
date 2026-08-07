#!/bin/bash

set -e


module load rocm 

echo "compiling"

hipcc --offload-arch=gfx942 tiled_version.cpp -O3 -o tiled
hipcc --offload-arch=gfx942 register_blocking.cpp -O3 -o reg_block
hipcc --offload-arch=gfx942 2d_register_blocking.cpp -O3 -o 2d_register_blocking
hipcc --offload-arch=gfx942 -O3 rocblas_matmul.cpp -lrocblas -o rocblas_matmul

echo
echo "===== Tiled ====="
./tiled

echo
echo "===== Register Blocking ====="
./reg_block

echo
echo "===== 2D Register Blocking ====="
./2d_register_blocking

echo
echo "===== rocBLAS ====="
./rocblas_matmul