"""
HNSW (Hierarchical Navigable Small World) is a graph-based approximate nearest neighbor search algorithm. It is designed to efficiently find the nearest neighbors of a query point in high-dimensional spaces. HNSW constructs a multi-layer graph where each layer represents a different level of granularity, allowing for fast navigation through the graph to find approximate nearest neighbors.

FAISS (Facebook AI Similarity Search) is an open-source library developed by Facebook AI Research that provides efficient similarity search and clustering of dense vectors. It implements various algorithms, including HNSW, to perform approximate nearest neighbor searches on large datasets.

In this example, we will compare the performance of an exact search index and an HNSW index using FAISS. We will create a dataset of 10,000 random vectors with 128 dimensions and perform a search for the nearest neighbors of a random query vector. We will measure the time taken for both the exact search and the HNSW search, as well as the overlap of results between the two methods. The goal is to demonstrate the speed improvement of HNSW over exact search while also highlighting the trade-off in terms of result accuracy (overlap percentage).
"""


import numpy as np
import faiss
import time

# Create sample data - 10,000 vectors with 128 dimensions
dimension = 128
num_vectors = 10000
vectors = np.random.random((num_vectors, dimension)).astype('float32')
query = np.random.random((1, dimension)).astype('float32')

# Exact search index
exact_index = faiss.IndexFlatL2(dimension)
exact_index.add(vectors)

# HNSW index (approximate but faster)
hnsw_index = faiss.IndexHNSWFlat(dimension, 32) # 32 connections per node
hnsw_index.add(vectors)

# Compare search times
start_time = time.time()
exact_D, exact_I = exact_index.search(query, k=10) # Search for 10 nearest neighbors
exact_time = time.time() - start_time


start_time = time.time()
hnsw_D, hnsw_I = hnsw_index.search(query, k=10)
hnsw_time = time.time() - start_time


# Calculate overlap (how many of the same results were found)
overlap = len(set(exact_I[0]).intersection(set(hnsw_I[0])))
overlap_percentage = overlap * 100 / 10
print(f"Exact search time: {exact_time:.6f} seconds")
print(f"HNSW search time: {hnsw_time:.6f} seconds")
print(f"Speed improvement: {exact_time/hnsw_time:.2f}x faster")
print(f"Result overlap: {overlap_percentage:.1f}%")


# Running this code typically produces results like:
# Exact search time: 0.003210 seconds
# HNSW search time: 0.000412 seconds
# Speed improvement: 7.79x faster
# Result overlap: 90.0%