#!/bin/bash

set -e

OUTPUT="../results/results.csv"

# Create CSV header
echo "Kernel,TILE_SIZE,REG_M,REG_N,Time_ms" > "$OUTPUT"

#######################################
# 2D Register Blocking
#######################################

echo "===== 2D Register Blocking ====="

for TILE in 8 16 32
do
    for RM in 1 2 4
    do
        for RN in 1 2 4
        do
            echo "----------------------------------"
            echo "TILE=$TILE REG_M=$RM REG_N=$RN"

            hipcc \
                --offload-arch=gfx942 \
                -O3 \
                -DTILE_SIZE=$TILE \
                -DREG_M=$RM \
                -DREG_N=$RN \
                ../src/2d_register_blocking.cpp \
                -o kernel_exec

            TIME=$(./kernel_exec | grep "Kernel time" | awk '{print $3}')

            echo "2D,$TILE,$RM,$RN,$TIME" >> "$OUTPUT"
        done
    done
done

#######################################
# Tiled
#######################################

echo "===== Tiled ====="

for TILE in 8 16 32 64
do
    echo "----------------------------------"
    echo "TILE=$TILE"

    hipcc \
        --offload-arch=gfx942 \
        -O3 \
        -DTILE_SIZE=$TILE \
        ../src/tiled_version.cpp \
        -o kernel_exec

    TIME=$(./kernel_exec | grep "Kernel time" | awk '{print $3}')

    echo "Tiled,$TILE,-,-,$TIME" >> "$OUTPUT"
done

#######################################
# 1D Register Blocking
#######################################

echo "===== Register Blocking ====="

for TILE in 8 16 32
do
    for RM in 1 2 4
    do
        echo "----------------------------------"
        echo "TILE=$TILE REG_M=$RM"

        hipcc \
            --offload-arch=gfx942 \
            -O3 \
            -DTILE_SIZE=$TILE \
            -DREG_M=$RM \
            ../src/register_blocking.cpp \
            -o kernel_exec

        TIME=$(./kernel_exec | grep "Kernel time" | awk '{print $3}')

        echo "1D,$TILE,$RM,-,$TIME" >> "$OUTPUT"
    done
done

#######################################
# Vectorized Loads
#######################################

echo "===== Vectorized Loads ====="

for TILE in 8 16 32
do
    echo "----------------------------------"
    echo "TILE=$TILE"

    hipcc \
        --offload-arch=gfx942 \
        -O3 \
        -DTILE_SIZE=$TILE \
        ../src/new_kernel.cpp \
        -o kernel_exec

    TIME=$(./kernel_exec | grep "Kernel time" | awk '{print $3}')

    echo "Vectorized,$TILE,-,-,$TIME" >> "$OUTPUT"
done

echo
echo "Results saved to $OUTPUT"
