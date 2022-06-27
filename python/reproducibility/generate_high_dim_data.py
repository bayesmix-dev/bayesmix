#!/usr/bin/env python

# Generate synthesized high-dimensional dataset

import numpy as np
import os

# Initialize settings
rng_seed = 20201124
dim = 4
points_per_clust = 5000

# Initialize mean and covariance matrix
mean = np.array(dim*[2])
cov = np.eye(dim)

# Generate dataset for two clusters
rng = np.random.default_rng(rng_seed)
samples1 = rng.multivariate_normal(mean=+mean, cov=cov, size=points_per_clust)
samples2 = rng.multivariate_normal(mean=-mean, cov=cov, size=points_per_clust)
samples = np.vstack((samples1, samples2))

# Save joined datasets to file
output_folder = 'csv'
os.makedirs(output_folder, exist_ok=True)
file = os.path.join(output_folder, 'highdim.csv')
np.savetxt(file, samples, delimiter=',', fmt="%.5f")
