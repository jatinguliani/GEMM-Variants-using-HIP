#!/bin/bash

set -e

OUTPUT="../results/scaling_results.csv"

echo "Kernel,MatrixSize,TileSize,REG_M,REG_N,Time_ms" > "$OUTPUT"

#######################################
# Tiled
#######################################

for S in 512 1024 2048 4096 8192 16384
do
    echo "----------------------------------"
    echo "size=$S"

    hipcc \
        --offload-arch=gfx942 \
        -O3 \
        -DTILE_SIZE=32 \
        -DM=$S \
        -DN=$S \
        -DK=$S \
        ../src/tiled_version.cpp \
        -o kernel_exec

    TIME=$(./kernel_exec | grep "Kernel time" | awk '{print $3}')

    echo "Tiled,$S,32,-,-,$TIME" >> "$OUTPUT"
done

#######################################
# 2D Register Blocking
#######################################

for S in 512 1024 2048 4096 8192 16384
do
    echo "----------------------------------"
    echo "size=$S"

    hipcc \
        --offload-arch=gfx942 \
        -O3 \
        -DTILE_SIZE=32 \
        -DM=$S \
        -DN=$S \
        -DK=$S \
        -DREG_M=4 \
        -DREG_N=1 \
        ../src/2d_register_blocking.cpp \
        -o kernel_exec

    TIME=$(./kernel_exec | grep "Kernel time" | awk '{print $3}')

    echo "2D Register Blocking,$S,32,4,1,$TIME" >> "$OUTPUT"
done

#######################################
# Vectorized
#######################################

for S in 512 1024 2048 4096 8192 16384
do
    echo "----------------------------------"
    echo "size=$S"

    hipcc \
        --offload-arch=gfx942 \
        -O3 \
        -DTILE_SIZE=16 \
        -DM=$S \
        -DN=$S \
        -DK=$S \
        ../src/new_kernel.cpp \
        -o kernel_exec

    TIME=$(./kernel_exec | grep "Kernel time" | awk '{print $3}')

    echo "Vectorized,$S,16,-,-,$TIME" >> "$OUTPUT"
done

echo
echo "Results saved to $OUTPUT"
