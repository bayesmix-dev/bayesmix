import matplotlib.pyplot as plt
import numpy as np


# paths needed
DATA_PATH = "../resources/csv/in/multivar/data_1000_80_v2.csv"
CLUSTER_PATH = "../resources/csv/out/multivar/nclusters_binder_1000_80_v2.csv"
FIG_PATH = "../resources/csv/out/multivar/plot_1000_80_v2.png"
COLOR = ['r', 'g', 'b', 'c', 'm', 'y']

def main():
    data = open(DATA_PATH)
    clusters = np.genfromtxt(CLUSTER_PATH, delimiter=',')
    x, y = [], []
    for line in data:
        z = line.split()
        x.append(float(z[0]))
        y.append(float(z[1]))

    fig = plt.figure(figsize=(21, 7))
    ax1 = fig.add_subplot()
    colors=compute_colors(clusters)
    for i in range(len(x)):
        ax1.plot(x[i], y[i], linestyle='None', marker='o', color=colors[i] , markersize = 10.0)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Cluster estimate for a multivariate case\nBinder Loss, Kup = 7')
    fig.show()
    fig.savefig(FIG_PATH, format='png')

def compute_colors(clusters):
    colors = []
    for c in clusters:
        colors.append(COLOR[int(c)-1])
    print(len(colors))
    return colors

if __name__ == '__main__':
    main()