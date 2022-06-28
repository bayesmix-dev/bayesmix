import os
import subprocess

HERE = os.path.dirname(os.path.realpath(__file__))


def run_shell(cmd, flush_startswith=None, cwd=None):
    proc = subprocess.Popen(
            cmd.split(),
            bufsize=1,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=os.environ,
            universal_newlines=True,
            cwd=cwd)

    while proc.poll() is None:
        if proc.stdout is not None:
            line = proc.stdout.readline()
            line = line.strip()
            if flush_startswith and \
                    line.startswith(flush_startswith):
                print("\r{0}".format(line), end=' ', flush=True)
            else:
                print("{0}".format(line))


def get_env_file():
    return os.path.join(HERE, ".env")
