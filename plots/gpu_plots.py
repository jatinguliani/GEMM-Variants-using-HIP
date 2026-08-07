import matplotlib.pyplot as plt
import pandas as pd

# Matrix size vs GFLOPS
def plot_gflops(matrix_size, time, title):
    gflops = []

    for n, t in zip(matrix_size, time):
        flops = 2 * n * n * n
        gflops.append(flops / (t * 1e6))

    plt.figure(figsize=(6,4))
    plt.plot(matrix_size, gflops, marker='o', linewidth=2)
    plt.title(title)
    plt.xlabel("Matrix Size")
    plt.ylabel("GFLOPS")
    plt.grid(True)
    plt.show()


# Heatmap for 2D Register Blocking
def plot_heatmap(csv_file, tile):
    df = pd.read_csv(csv_file)

    data = df[
        (df["Kernel"] == "2D") &
        (df["TILE_SIZE"] == tile)
    ]

    heatmap = data.pivot(
        index="REG_M",
        columns="REG_N",
        values="Time_ms"
    )

    plt.figure(figsize=(5,4))

    plt.imshow(heatmap, cmap="viridis", aspect="auto")

    plt.xticks(range(len(heatmap.columns)), heatmap.columns)
    plt.yticks(range(len(heatmap.index)), heatmap.index)

    plt.xlabel("REG_N")
    plt.ylabel("REG_M")
    plt.title(f"Tile Size = {tile}")

    plt.colorbar(label="Execution Time (ms)")

    for i in range(len(heatmap.index)):
        for j in range(len(heatmap.columns)):
            plt.text(
                j, i,
                f"{heatmap.iloc[i,j]:.2f}",
                ha="center",
                va="center",
                color="white"
            )

    plt.show()


# function calls
# naive
matrix_size = [512, 1024, 2048]
time = [284.043, 5720.08, 56411.5]
plot_gflops(matrix_size, time, "Naive GEMM")

# 2D register
plot_heatmap("results.csv", 16)
plot_heatmap("results.csv", 32)

#1D register 
reg_m = [1, 2, 4]
tile8  = [85.1348, 157.766, 296.542]
tile16 = [65.0373, 58.2433, 51.8274]
tile32 = [66.8237, 51.4518, 43.4962]

# Convert to GFLOPS (assuming TILE_SIZE = 4096)
N = 4096

def to_gflops(times):
    return [(2 * N**3) / (t * 1e6) for t in times]

tile8_gflops  = to_gflops(tile8)
tile16_gflops = to_gflops(tile16)
tile32_gflops = to_gflops(tile32)

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].plot(reg_m, tile8, marker='o', label='Tile 8')
ax[0].plot(reg_m, tile16, marker='o', label='Tile 16')
ax[0].plot(reg_m, tile32, marker='o', label='Tile 32')

ax[0].set_title("Execution Time")
ax[0].set_xlabel("REG_M")
ax[0].set_ylabel("Time (ms)")
ax[0].grid(True)
ax[0].legend()

ax[1].plot(reg_m, tile8_gflops, marker='o', label='Tile 8')
ax[1].plot(reg_m, tile16_gflops, marker='o', label='Tile 16')
ax[1].plot(reg_m, tile32_gflops, marker='o', label='Tile 32')

ax[1].set_title("Performance")
ax[1].set_xlabel("REG_M")
ax[1].set_ylabel("GFLOPS")
ax[1].grid(True)
ax[1].legend()

plt.suptitle("1D Register Blocking Autotuning")
plt.tight_layout()

plt.savefig("1d_register_blocking.png", dpi=300)
plt.show()


# Vectorized Loading
tile_size = [8, 16, 32]
time = [232.059, 126.908, 240.107]
N = 4096

gflops = []
for t in time:
    gflops.append((2 * N**3) / (t * 1e6))

fig, ax = plt.subplots(1, 2, figsize=(12,5))

ax[0].plot(tile_size, time, marker='o', linewidth=2)
ax[0].set_title("Execution Time")
ax[0].set_xlabel("Tile Size")
ax[0].set_ylabel("Time (ms)")
ax[0].grid(True)

ax[1].plot(tile_size, gflops, marker='o', linewidth=2)
ax[1].set_title("Performance")
ax[1].set_xlabel("Tile Size")
ax[1].set_ylabel("GFLOPS")
ax[1].grid(True)

plt.suptitle("Vectorized Loading Autotuning")
plt.tight_layout()
plt.savefig("vectorized_autotune.png", dpi=300)
plt.show()

# best configuration for each method vs time vs G flops
import matplotlib.pyplot as plt

matrix_size = [512, 1024, 2048, 4096, 8192, 16384]

# Best Tiled (Tile = 32)
tiled_time = [0.592161, 1.53856, 8.82919, 65.7401, 539.186, 4506.2]

# Best 2D Register Blocking (Tile = 32, REG_M = 4, REG_N = 1)
rb2d_time = [0.10368, 0.698163, 5.40027, 42.6662, 345.745, 2832.58]

# Best Vectorized (Tile = 16)
vector_time = [0.259359, 1.98657, 15.846, 126.568, 1011.96, 8087.5]


def get_gflops(size, time):
    gflops = []

    for n, t in zip(size, time):
        flops = 2 * n * n * n
        gflops.append(flops / (t * 1e6))

    return gflops

tiled_gflops = get_gflops(matrix_size, tiled_time)
rb2d_gflops = get_gflops(matrix_size, rb2d_time)
vector_gflops = get_gflops(matrix_size, vector_time)

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].plot(matrix_size, tiled_time,
           marker='o',
           label='Tiled (Tile=32)')

ax[0].plot(matrix_size, rb2d_time,
           marker='s',
           label='2D RB (Tile=32, REG_M=4, REG_N=1)')

ax[0].plot(matrix_size, vector_time,
           marker='^',
           label='Vectorized (Tile=16)')

ax[0].set_title("Execution Time")
ax[0].set_xlabel("Matrix Size")
ax[0].set_ylabel("Time (ms)")
ax[0].grid(True)
ax[0].legend()

ax[1].plot(matrix_size, tiled_gflops,
           marker='o',
           label='Tiled (Tile=32)')

ax[1].plot(matrix_size, rb2d_gflops,
           marker='s',
           label='2D RB (Tile=32, REG_M=4, REG_N=1)')

ax[1].plot(matrix_size, vector_gflops,
           marker='^',
           label='Vectorized (Tile=16)')

ax[1].set_title("Performance")
ax[1].set_xlabel("Matrix Size")
ax[1].set_ylabel("GFLOPS")
ax[1].grid(True)
ax[1].legend()

plt.tight_layout()
plt.savefig("best_comparison.png", dpi=300)
plt.show()