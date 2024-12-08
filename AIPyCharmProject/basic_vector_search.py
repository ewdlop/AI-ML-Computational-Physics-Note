import numpy as np
import faiss #linux and mac only

#use wsl to test or something

# Parameters
d = 64  # dimension
nb = 100000  # database size
nq = 10000  # number of queries

# Generate the database and query vectors
np.random.seed(1234)  # make reproducible
xb = np.random.random((nb, d)).astype('float32')
xq = np.random.random((nq, d)).astype('float32')

# Assign unique IDs to the first dimension
xb[:, 0] = np.arange(nb)
xq[:, 0] = np.arange(nq)

# Normalize the vectors (excluding the first dimension which is the unique ID)
xb[:, 1:] = xb[:, 1:] / np.linalg.norm(xb[:, 1:], axis=1, keepdims=True)
xq[:, 1:] = xq[:, 1:] / np.linalg.norm(xq[:, 1:], axis=1, keepdims=True)

# Build the index
index = faiss.IndexFlatIP(d)  # Use Inner Product (IP) for normalized vectors
index.add(xb)  # Add vectors to the index

# Perform the search
k = 5  # we want to find the 5 nearest neighbors
D, I = index.search(xq, k)  # D = distances, I = indices of nearest neighbors

# Output the results
print("Distances to nearest neighbors:\n", D)
print("Indices of nearest neighbors:\n", I)

