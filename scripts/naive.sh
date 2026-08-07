OUTPUT="naive_results.csv"

echo "Kernel,MatrixSize,Time_ms" > "$OUTPUT"

for SIZE in 512 1024 2048 8192
do
    hipcc \
        -O3 \
        -DM=$SIZE \
        -DN=$SIZE \
        -DK=$SIZE \
        naive.cpp \
        -o naive

    TIME=$(./naive | grep "Execution Time" | awk '{print $4}')

    echo "Naive,$SIZE,$TIME" >> "$OUTPUT"
done

echo "Results saved to $OUTPUT"