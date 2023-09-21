import os
import shutil
import subprocess
import numpy as np

from dotenv import load_dotenv
from tempfile import TemporaryDirectory
from pathlib import Path

from .shell_utils import get_env_file, run_shell
from .io_utils import maybe_print_proto_to_file, read_many_protos_from_file




def _get_filenames(outdir):
    data = os.path.join(outdir, 'data.csv')
    dens_grid = os.path.join(outdir, 'dens_grid.csv')
    n_clus = os.path.join(outdir, 'n_clus.csv')
    clus = os.path.join(outdir, 'clus.csv')
    best_clus = os.path.join(outdir, 'best_clus.csv')
    return data, dens_grid, n_clus, clus, best_clus


def run_mcmc(
        hierarchy: str,
        mixing: str,
        data: np.array,
        hier_params: str,
        mix_params: str,
        algo_params: str,
        dens_grid: np.array = None,
        out_dir: str = None,
        return_clusters: bool = True,
        return_best_clus: bool = True,
        return_num_clusters: bool = True,
        return_chains: bool = False):
    """
    Run the MCMC sampling by calling the Bayesmix executable from a subprocess.

    Parameters
    ----------

    hierarchy: str.
        The id of the hyerarchy. Must be one of the 'Name' in
        http://bayesmix.readthedocs.io/en/latest/protos.html#hierarchy_id.proto
    mixing: str.
        The id of the mixing. Must be one of the 'Name' in
        http://bayesmix.readthedocs.io/en/latest/protos.html#mixing_id.proto
    data: np.array of shape (n_samples, n_dim).
        Observations on which to fit the model.
    hier_params: str.
        A text string containing the hyperparameters of the hierarchy or
        a file name where the hyperparameters are stored. A protobuf message of
        the corresponding type will be created and populated with the
        parameters. See the file hierarchy_prior.proto for the corresponding
        message.
    mix_params: str.
        A text string containing the hyperparameters of the mixing or
        a file name where the hyperparameters are stored. A protobuf message of
        the corresponding type will be created and populated with the
        parameters. See the file mixing_prior.proto for the corresponding
        message.
    algo_params: str.
        A text string containing the hyperparameters of the algorithm or
        a file name where the hyperparameters are stored.
        See the file algorithm_params.proto for the corresponding message.
    dens_grid: np.array of shape (n_dens_grid_points,).
        Rpoints where to evaluate the density.
        If None, the density will not be evaluated.
    out_dir: str.
        If not None, where to store the output. If None, a temporary directory
        will be created and destroyed after the sampling is finished.
    return_clusters: bool.
        If True, returns the chain of the cluster allocations.
    return_best_clus: bool.
        If True, returns the best cluster allocation obtained
        by minimizing the Binder loss function over the visited partitions
        during the MCMC sampling.
    return_num_clusters: bool.
        If True, returns the chain of the number of clusters.
    return_chains: bool.
        If True, the whole MCMC chain of the parameters is returned.


    Returns
    -------

    eval_dens: np.array of shape (n_samples, n_dens_grid_points).
        For each iteration, the mixture density evaluated at the points in
        dens_grid. None if eval_dens is False.
    n_clus: np.array of shape (n_samples,).
        The number of clusters for each iteration. None if return_num_clusters
        is False.
    clus_chain: np.array shape (n_samples, n_data).
        The cluster allocation for each iteration. None if return_clusters is
        False.
    best_clus: np.array of shape (n_data,).
        The best clustering obtained by minimizing Binder's loss function. None
        if return_best_clus is False.
    mcmc_chain: list of AlgorithmState objects
        The whole MCMC chain as a list of AlgorithmState protobuf objects. None
        if return_chains is False
        See https://bayesmix.readthedocs.io/en/latest/protos.html#algorithm_state.proto
        for furhter details on the AlgorithmState object
    """
    if return_chains is not None:
        from .proto.algorithm_state_pb2 import AlgorithmState

    load_dotenv(get_env_file())
    BAYESMIX_EXE = os.getenv("BAYESMIX_EXE")
    if BAYESMIX_EXE is None:
        raise ValueError("BAYESMIX_EXE environment variable not set")

    RUN_CMD = BAYESMIX_EXE + """ --algo-params-file {0} --hier-type {1} \
                                 --hier-args {2} --mix-type {3} \
                                 --mix-args {4} --coll-name {5} \
                                 --data-file {6} --grid-file {7} \
                                 --dens-file {8} --n-cl-file {9} \
                                 --clus-file {10} --best-clus-file {11}"""


    if out_dir is None:
        out_dir = TemporaryDirectory().name
        os.makedirs(out_dir, exist_ok=True)
        remove_out_dir = True
    else:
        remove_out_dir = False

    data_file, dens_grid_file, nclus_file, clus_file, best_clus_file = \
        _get_filenames(out_dir)
    hier_params_file = maybe_print_proto_to_file(hier_params, "hier_params",
                                                 out_dir)
    mix_params_file = maybe_print_proto_to_file(mix_params, "mix_params", out_dir)
    algo_params_file = maybe_print_proto_to_file(algo_params, "algo_params",
                                                 out_dir)
    eval_dens_file = os.path.join(out_dir, "eval_dens.csv")

    np.savetxt(data_file, data, fmt='%1.5f', delimiter=',')
    if dens_grid is None:
        dens_grid_file = '\"\"'
        eval_dens_file = '\"\"'
    else:
        np.savetxt(dens_grid_file, dens_grid, fmt='%1.5f', delimiter=',')

    if not return_clusters:
        clus_file = '\"\"'

    if not return_num_clusters:
        nclus_file = '\"\"'

    if not return_best_clus:
        best_clus_file = '\"\"'

    if return_chains:
        coll_name = os.path.join(out_dir, "chains.recordio")
    else:
        coll_name = "memory"

    cmd = RUN_CMD.format(
        algo_params_file,
        hierarchy, hier_params_file,
        mixing, mix_params_file,
        coll_name,
        data_file,
        dens_grid_file,
        eval_dens_file,
        nclus_file,
        clus_file,
        best_clus_file)

    try:
        run_shell(cmd, flush_startswith=("[>", "[="))
    except OSError as e:
        msg = 'Failed with error {}\n'.format(str(e))
        if remove_out_dir:
            shutil.rmtree(out_dir, ignore_errors=True)
        raise RuntimeError(msg) from e

    eval_dens = None
    if dens_grid is not None:
        eval_dens = np.loadtxt(eval_dens_file, delimiter=',')

    nclus = None
    if return_num_clusters:
        nclus = np.loadtxt(nclus_file, delimiter=',')

    clus = None
    if return_clusters:
        clus = np.loadtxt(clus_file, delimiter=',')

    best_clus = None
    if return_best_clus:
        best_clus = np.loadtxt(best_clus_file, delimiter=',')

    mcmc_chains = None
    if return_chains:
        mcmc_chains = read_many_protos_from_file(coll_name, AlgorithmState)

    if remove_out_dir:
        shutil.rmtree(out_dir, ignore_errors=True)

    return eval_dens, nclus, clus, best_clus, mcmc_chains
