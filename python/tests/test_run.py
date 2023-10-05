import numpy as np
from bayesmixpy import run_mcmc

DP_PARAMS = """
fixed_value {
    totalmass: 1.0
}
"""

GO_PARAMS = """
fixed_values {
    mean: 0.0
    var_scaling: 0.1
    shape: 2.0
    scale: 2.0
}
"""

ALGO_PARAMS = """
    algo_id: "Neal2"
    rng_seed: 20201124
    iterations: 10
    burnin: 5
    init_num_clusters: 3
"""

def get_data():
    np.random.seed(0)
    return np.random.normal(0, 1, size=(10,))


def get_grid():
    return np.linspace(-5, 5, 10)


def test_run_mcmc():
    data = get_data()
    grid = get_grid()

    eval_dens, nclus, clus, best_clus, chains = run_mcmc(
            "NNIG", "DP", data, GO_PARAMS, DP_PARAMS,
            ALGO_PARAMS, grid, return_clusters=False,
            return_num_clusters=False, return_best_clus=False,
            return_chains=False)

    assert eval_dens.shape[0] == 5
    assert eval_dens.shape[1] == len(grid)
    assert nclus is None
    assert clus is None
    assert best_clus is None
    assert chains is None

    eval_dens, nclus, clus, best_clus, chains = run_mcmc(
            "NNIG", "DP", data, GO_PARAMS, DP_PARAMS,
            ALGO_PARAMS, None, return_clusters=False,
            return_num_clusters=True, return_best_clus=False,
            return_chains=False)

    assert eval_dens is None
    assert nclus is not None
    assert len(nclus) == 5
    assert clus is None
    assert best_clus is None
    assert chains is None

    eval_dens, nclus, clus, best_clus, chains = run_mcmc(
            "NNIG", "DP", data, GO_PARAMS, DP_PARAMS,
            ALGO_PARAMS, None, return_clusters=True,
            return_num_clusters=False, return_best_clus=False,
            return_chains=False)
    assert chains is None

    assert eval_dens is None
    assert nclus is None
    assert clus is not None
    assert clus.shape[0] == 5
    assert clus.shape[1] == len(data)
    assert best_clus is None
    assert chains is None

    eval_dens, nclus, clus, best_clus, chains = run_mcmc(
            "NNIG", "DP", data, GO_PARAMS, DP_PARAMS,
            ALGO_PARAMS, None, return_clusters=False,
            return_num_clusters=False, return_best_clus=True,
            return_chains=False)

    assert eval_dens is None
    assert nclus is None
    assert clus is None
    assert best_clus is not None
    assert len(best_clus) == len(data)
    assert chains is None


    eval_dens, nclus, clus, best_clus, chains = run_mcmc(
            "NNIG", "DP", data, GO_PARAMS, DP_PARAMS,
            ALGO_PARAMS, None, return_clusters=False,
            return_num_clusters=False, return_best_clus=False,
            return_chains=True)
    assert eval_dens is None
    assert nclus is None
    assert clus is None
    assert best_clus is None
    assert len(chains) == 5
