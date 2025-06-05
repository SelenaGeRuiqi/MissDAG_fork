import numpy as np

# Load and inspect a .npy file
matrix = np.load('result_npy/s100_mar_v20_e5_m10_seed777/ground_truth_cpdag_adj.npy')
print(f"Shape: {matrix.shape}")
print(f"Data type: {matrix.dtype}")
print("Matrix:")
print(matrix)