import os
import shutil
import subprocess
import numpy as np

from tempfile import TemporaryDirectory
from pathlib import Path

from .shell_utils import run_shell


def _is_file(a: str):
    p = Path(a)
    return p.exists() and p.is_file()


def _maybe_print_to_file(maybe_proto: str,
                         proto_name: str = None,
                         out_dir: str = None):
    """If maybe_proto is a file, returns the file name.
    If maybe_proto is a string representing a message, prints the message to
    a file and returns the file name.
    """
    if _is_file(maybe_proto):
        return maybe_proto

    proto_file = os.path.join(out_dir, proto_name + ".asciipb")
    with open(proto_file, "w") as f:
        print(maybe_proto, file=f)

    return proto_file



def _get_filenames(outdir):
    data = os.path.join(outdir, 'data.csv')
    dens_grid = os.path.join(outdir, 'dens_grid.csv')
    n_clus = os.path.join(outdir, 'n_clus.csv')
    clus = os.path.join(outdir, 'clus.csv')
    return data, dens_grid, n_clus, clus


def run_mcmc(
        hierarchy: str,
        mixing: str,
        data: np.array,
        hier_params: str,
        mix_params: str,
        algo_params: str,
        dens_grid: np.array = None,
        out_dir: str = None):

    """
    Run the MCMC sampling by calling the Bayesmix executable from a subprocess.
    Arguments
    ---------
    hierarchy: the id of the hyerarchy. Must be one of ["NNIG", "NNW", "LinRegUni"]
    mixing: the id of the mixing. Must be one of ["DP", "PY", "LogSB", "TruncSB"]
    data: a numpy array of shape (n_samples, dim)
    hier_params: a text string containing the hyperparameters of the hierarchy or
        a file name where the hyperparameters are stored. A protobuf message of the
        corresponding type will be created and populated with the parameters.
        See the file hierarchy_prior.proto for the corresponding message.
    mix_params: a text string containing the hyperparameters of the mixing or
        a file name where the hyperparameters are stored. A protobuf message of the
        corresponding type will be created and populated with the parameters.
        See the file mixing_prior.proto for the corresponding message.
    algo_params: a text string containing the hyperparameters of the algorithm or
        a file name where the hyperparameters are stored.
        See the file algorithm_params.proto for the corresponding message.
    dens_grid: a numpy array of shape (n_dens_grid_points,): points where to evaluate
        the density. If None, the density will not be evaluated.
    out_dir: if not None, where to store the output. If None, a temporary directory
        will be created and destroyed after the sampling is finished.

    Returns
    -------
    eval_dens: a numpy array of shape (n_samples, n_dens_grid_points):
        for each iteration, the mixture density evaluated at the points in dens_grid.
    n_clus: a numpy array of shape (n_samples,): the number of clusters for each iteration.
    clus: the best clustering obtained by minimizing Binder's loss function.
    """

    BAYESMIX_EXE = os.environ.get("BAYESMIX_EXE", default="")
    RUN_CMD = BAYESMIX_EXE + " {0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10}"


    if out_dir is None:
        out_dir = TemporaryDirectory().name
        os.makedirs(out_dir, exist_ok=True)
        remove_out_dir = True
    else:
        remove_out_dir = False

    data_file, dens_grid_file, nclus_file, clus_file = _get_filenames(out_dir)
    np.savetxt(data_file, data, delimiter=',')
    np.savetxt(dens_grid_file, dens_grid, delimiter=',')

    hier_params_file = _maybe_print_to_file(hier_params, "hier_params", out_dir)
    mix_params_file = _maybe_print_to_file(mix_params, "mix_params", out_dir)
    algo_params_file = _maybe_print_to_file(algo_params, "algo_params", out_dir)

    eval_dens_file = os.path.join(out_dir, "eval_dens.csv")

    cmd = RUN_CMD.format(
        algo_params_file,
        hierarchy, hier_params_file,
        mixing, mix_params_file,
        'memory',
        data_file,
        dens_grid_file,
        eval_dens_file,
        nclus_file,
        clus_file)

    try:
        run_shell(cmd, flush_startswith=("[>", "[="))
    except OSError as e:
        msg = 'Failed with error {}\n'.format(str(e))
        if remove_out_dir:
            shutil.rmtree(out_dir, ignore_errors=True)
        raise RuntimeError(msg) from e

    eval_dens = np.loadtxt(eval_dens_file, delimiter=',')
    nclus = np.loadtxt(nclus_file, delimiter=',')
    clus = np.loadtxt(clus_file, delimiter=',')

    if remove_out_dir:
        shutil.rmtree(out_dir, ignore_errors=True)

    return eval_dens, nclus, clus
