import glob
import os
import platform
import shutil
import sys
import subprocess



def maybe_build_tbb():
    """Build tbb. This function is taken from
    https://github.com/stan-dev/pystan/blob/develop/setup.py"""

    
    stan_math_lib = os.path.abspath(os.path.join(os.path.dirname(
        __file__), 'lib', 'math', 'lib'))

    tbb_dir = os.path.join(stan_math_lib, 'tbb')
    tbb_dir = os.path.abspath(tbb_dir)
    if os.path.exists(tbb_dir):
        return

    make = 'make' if platform.system() != 'Windows' else 'mingw32-make'
    cmd = [make]

    tbb_root = os.path.join(stan_math_lib, 'tbb_2019_U8').replace("\\", "/")

    cmd.extend(['-C', tbb_root])
    cmd.append('tbb_build_dir={}'.format(stan_math_lib))
    cmd.append('tbb_build_prefix=tbb')
    cmd.append('tbb_root={}'.format(tbb_root))

    cmd.append('stdver=c++14')

    cmd.append('compiler=gcc')

    cwd = os.path.abspath(os.path.dirname(__file__))

    subprocess.check_call(cmd, cwd=cwd)

    tbb_debug = os.path.join(stan_math_lib, "tbb_debug")
    tbb_release = os.path.join(stan_math_lib, "tbb_release")
    tbb_dir = os.path.join(stan_math_lib, "tbb")

    if not os.path.exists(tbb_dir):
        os.makedirs(tbb_dir)

    if os.path.exists(tbb_debug):
        shutil.rmtree(tbb_debug)

    shutil.move(os.path.join(tbb_root, 'include'), tbb_dir)
    shutil.rmtree(tbb_root)

    for name in os.listdir(tbb_release):
        srcname = os.path.join(tbb_release, name)
        dstname = os.path.join(tbb_dir, name)
        shutil.move(srcname, dstname)

    if os.path.exists(tbb_release):
        shutil.rmtree(tbb_release)


if __name__ == "__main__":
    maybe_build_tbb()