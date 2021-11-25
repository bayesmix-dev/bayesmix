import os
import shutil
import subprocess
import numpy as np

from tempfile import TemporaryDirectory
from pathlib import Path

from .shell_utils import run_shell


BAYESMIX_EXE = os.environ.get("BAYESMIX_EXE", default="")
RUN_CMD = BAYESMIX_EXE + " {0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10}"


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



def get_filenames(outdir):
    data = os.path.join(outdir, 'data.csv')
    dens_grid = os.path.join(outdir, 'dens_grid.csv')
    n_clus = os.path.join(outdir, 'n_clus.csv')
    clus = os.path.join(outdir, 'clus.csv')
    return data, dens_grid, n_clus, clus


def run_mcmc(
        hierarchy: str,
        mixing: str,
        data: np.array,
        dens_grid: np.array,
        hier_params: str,
        mix_params: str,
        algo_params: str,
        out_dir: str = None):

    if out_dir is None:
        out_dir = TemporaryDirectory().name
        os.makedirs(out_dir, exist_ok=True)
        remove_out_dir = True
    else:
        remove_out_dir = False

    data_file, dens_grid_file, nclus_file, clus_file = get_filenames(out_dir)
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
