import numpy as np

# Load your DDS file
dds = np.load("data/dds_results/test_000.npy")

print(f"DDS shape: {dds.shape}")
print(f"DDS dtype: {dds.dtype}")
print(f"Sample values:\n{dds[0:5]}")
print(f"Number of zeros: {np.sum(dds == 0)}")
print(f"Total elements: {dds.size}")