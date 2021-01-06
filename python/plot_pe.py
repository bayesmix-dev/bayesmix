import numpy as np
import matplotlib.pyplot as plt
import subprocess
import sys

# paths needed
DATA_PATH = "../resources/csv/in/data.csv"
CLUSTER_PATH = "../resources/csv/out/clusters_pe_vi.csv"
PLOT_PATH = "../resources/csv/plot/clusters_pe_vi.png"
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

    colors = ["#000000", "#ff6f31", "#d2d68a", "#474943", "#6f8969", "#d26267", "#227744", "#332255"]
    markers = ["o", "X", "p", "<", ">", 's', '^', '*', 'D']
    data = np.genfromtxt(DATA_PATH, delimiter='\n')
    clusters = np.genfromtxt(CLUSTER_PATH, delimiter=',')

    clusters_already_seen = []
    cluster_indexes = []

    fig = plt.figure(figsize=(21, 7))

    # group clusters by index.
    for index, cluster in enumerate(clusters):
        if cluster in clusters_already_seen:
            continue

        clusters_already_seen.append(cluster)
        new_indexes = []
        for other_index, other in enumerate(clusters[index:]):
            if other == cluster:
                new_indexes.append(other_index + index)
        cluster_indexes.append(new_indexes)

    ax1 = fig.add_subplot()
    ax1.set_title("Clustering point estimate with " + LOSS_TYPE + " Loss")
    # for cluster in clusters:
    for i, cluster_index in enumerate(cluster_indexes):
        ax1.plot([data[i] for i in cluster_index],
                 [0 for i in range(len(cluster_index))],
                 linestyle='none', marker=markers[i], c=colors[i], markersize=10)

    fig.show()
    fig.savefig(PLOT_PATH, format='png')
    # plt.show()


if __name__ == "__main__":
    # execute only if run as a script
    main()
