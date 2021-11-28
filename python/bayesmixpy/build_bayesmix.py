import os
import pathlib
import subprocess

from .shell_utils import run_shell

HERE = os.path.dirname(os.path.realpath(__file__))
path = pathlib.Path(HERE)
BAYESMIX_HOME = path.resolve().parents[1]


def build_bayesmix(nproc=1):
    print("Building the Bayesmix executable")
    build_dir = os.path.join(BAYESMIX_HOME, 'build')
    os.makedirs(build_dir, exist_ok=True)
    cmake_cmd = "cmake .. -DDISABLE_DOCS=TRUE -DDISABLE_BENCHMARKS=TRUE " + \
        "-DDISABLE_TESTS=TRUE"
    try:
        run_shell(cmake_cmd, cwd=build_dir)
    except subprocess.CalledProcessError as e:
        print(e)
        print("Some error has occurred while building Bayesmix. The library has not"
              " been installed!")
        return

    run_cmd = "make run -j{}".format(nproc)
    try:
        run_shell(run_cmd, cwd=build_dir)
    except subprocess.CalledProcessError as e:
        print(e)
        print("Some error has occurred while building Bayesmix. The library has not"
              " been installed!")
        return

    print("Bayesmix executable is in '{0}', \nexport the environment"
           " variable BAYESMIX_EXE={0}/{1}".format(build_dir, "run"))
    return True

if __name__ == '__main__':
    build_bayesmix(4)
