import os
import pathlib
import subprocess

HERE = os.path.dirname(os.path.realpath(__file__))
path = pathlib.Path(HERE)
BAYESMIX_HOME = path.resolve().parents[1]

def build_bayesmix(nproc):
    build_dir = os.path.join(BAYESMIX_HOME, 'build')
    os.makedirs(build_dir, exist_ok=True)
    cmake_cmd = "cmake .. -DDISABLE_DOCS=TRUE -DDISABLE_BENCHMARKS=TRUE " + \
        "-DDISABLE_TESTS=TRUE"
    subprocess.run(cmake_cmd.split(),cwd=build_dir)
    run_cmd = "make run -j{}".format(nproc)
    subprocess.run(run_cmd.split(), cwd=build_dir)
    print("Bayesmix executable is in '{0}', \nexport the environment"
           " variable BAYESMIX_HOME={0}".format(build_dir))


if __name__ == '__main__':
    build_bayesmix(4)
