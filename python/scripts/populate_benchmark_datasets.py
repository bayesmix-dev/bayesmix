import numpy as np
import os

multivariate_dims = [2, 4, 8]
N_BY_CLUS = 10
BASE_PATH = os.path.join("resources", "benchmarks", "datasets")
BASE_CHAIN_PATH = os.path.join("resources", "benchmarks", "chains")

if __name__ == '__main__':
    os.makedirs(BASE_PATH, exist_ok=True)
    os.makedirs(BASE_CHAIN_PATH, exist_ok=True)

    np.random.seed(2021)

    univ_y = np.concatenate(
        [np.random.normal(loc=-5, size=N_BY_CLUS),
         np.random.normal(loc=5, size=N_BY_CLUS)])

    fname = os.path.join(BASE_PATH, "univariate_gaussian.csv")
    np.savetxt(fname, univ_y, delimiter=',')

    for d in multivariate_dims:
        multiv_y = np.vstack(
            [np.random.normal(loc=-5, size=(N_BY_CLUS, d)),
             np.random.normal(loc=5, size=(N_BY_CLUS, d))])

        fname = os.path.join(
            BASE_PATH, "multi_gaussian_dim_{0}.csv".format(d))
        np.savetxt(fname, multiv_y, delimiter=',')
