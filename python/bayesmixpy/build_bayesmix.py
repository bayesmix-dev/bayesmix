import os
import pathlib
import subprocess

from .shell_utils import run_shell

HERE = os.path.dirname(os.path.realpath(__file__))
path = pathlib.Path(HERE)
BAYESMIX_HOME = path.resolve().parents[1]


def build_bayesmix(nproc=1, build_dirname="build"):
    """
    Builds the BayesMix executable. After the build, if no error has occurred,
    it prints out the path to the executable. Save the path into the environment
    variable BAYESMIX_EXE.

    Parameters
    ----------

    nproc : int
        Number of processes to use for parallel compilation.
    """
    print("Building the Bayesmix executable")
    build_dir = os.path.join(BAYESMIX_HOME, build_dirname)
    os.makedirs(build_dir, exist_ok=True)
    cmake_cmd = "cmake .. -DDISABLE_DOCS=TRUE -DDISABLE_BENCHMARKS=TRUE " + \
        "-DDISABLE_TESTS=TRUE -DCMAKE_BUILD_TYPE=Release"
    try:
        run_shell(cmake_cmd, cwd=build_dir)
    except subprocess.CalledProcessError as e:
        print(e)
        print("Some error has occurred while building Bayesmix. The library has not"
              " been installed!")
        return

    run_cmd = "make run_mcmc -j{}".format(nproc)
    try:
        run_shell(run_cmd, cwd=build_dir)
    except subprocess.CalledProcessError as e:
        print(e)
        print("Some error has occurred while building Bayesmix. The library has not"
              " been installed!")
        return

    print("Bayesmix executable is in '{0}', \nexport the environment"
           " variable BAYESMIX_EXE={0}/{1}".format(build_dir, "run_mcmc"))
    return True

if __name__ == '__main__':
    build_bayesmix(4)
