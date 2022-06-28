import os
import pathlib
import subprocess

from dotenv import set_key

from .shell_utils import get_env_file, run_shell

HERE = os.path.dirname(os.path.realpath(__file__))
path = pathlib.Path(HERE)
BAYESMIX_HOME = path.resolve().parents[1]


def set_bayesmix_env(run_path):
    env_file = get_env_file()
    if not os.path.exists(env_file):
        open(env_file, mode='a').close()

    set_key(env_file, "BAYESMIX_EXE", run_path)


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
    cmake_cmd = "cmake .. -DDISABLE_BENCHMARKS=TRUE -DDISABLE_TESTS=TRUE " + \
        "-DDISABLE_PLOTS=TRUE -DCMAKE_BUILD_TYPE=Release"
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

    set_bayesmix_env("{0}/{1}".format(build_dir, "run_mcmc"))
    return True

if __name__ == '__main__':
    build_bayesmix(4)
