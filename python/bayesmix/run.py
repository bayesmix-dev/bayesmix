import os
import shutil
import numpy as np

from tempfile import TemporaryDirectory


BAYESMIX_EXE = os.environ["BAYESMIX_EXE"]
RUN_CMD = BAYESMIX_EXE + "{0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10}"

def get_filenames(outdir):
    data = os.path.join(outdir, 'data.csv')
    dens_grid = os.path.join(outdir, 'dens_grid.csv')
    n_clus = os.path.join(outdir, 'n_clus.csv')
    clus = os.path.join(outdir, 'clus.csv')
    return data, dens_grid, n_clus, clus


def run_mcmc(
        hierarchy: str,
        mixing: str,
        algorithm: str,
        data: np.array,
        dens_grid: np.array,
        hier_params: str = None,
        mix_params: str = None,
        algo_params: str = None,
        n_iter: int = 2000,
        n_burn: int = 1000,
        n_thin: int = 1,
        out_dir: str = None):

    if out_dir is None:
        out_dir = TemporaryDirectory()
        remove_out_dir = True
    else:
        remove_out_dir = False

    data_file, dens_file, nclus_file, clus_file = get_filenames(out_dir)
    np.savetxt(data_file, data, delimiter=',')
    np.savetxt(dens_file, dens_grid, delimiter=',')

    # TODO: handle default hier_params, mix_params, algo_params!!

    cmd = RUN_CMD.format(
        hierarchy,
    )

    if remove_out_dir:
        shutil.rmtree(out_dir, ignore_errors=True)
    






