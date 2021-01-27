import numpy as np
import matplotlib.pyplot as plt
import subprocess
import sys

# paths needed
EPL_PATH = "../resources/csv/out/epl_vi.csv"
PLOT_PATH = "../resources/csv/plot/epl_vi.png"
LOSS_TYPE = "VI"


def run_pe():
    cmd = ["../build/run_pe",
           "../resources/csv/in/clusters2.csv",
           "../resources/csv/out/clusters_pe_vi.csv",
           "1"]

    subprocess.run(cmd, stdout=True)


def main():
    if len(sys.argv) > 1:
        run_pe()

    data = np.genfromtxt(EPL_PATH, delimiter=',')

    fig = plt.figure(figsize=(21, 7))
    ax1 = fig.add_subplot()
    # ax1.set_title("posterior loss  with " + LOSS_TYPE + " Loss")
    ax1.plot(range(1,len(data) + 1), [_ for _ in data])
    fig.show()
    fig.savefig(PLOT_PATH, format='png')
    # plt.show()


if __name__ == "__main__":
    # execute only if run as a script
    main()
