import numpy as np
data = np.load('npz/operational/sfc/regular/2025070312.npz')

for i in data.keys():
    print(f"{i}: {data[i].shape}, dtype={data[i].dtype}")