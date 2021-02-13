import matplotlib.pyplot as plt
import os
import subprocess
import time

# paths needed
CLUSTER_DIR = "../resources/csv/in/clusters"
PLOT_PATH = "../resources/csv/plot/times_col_constant.png"


def measure_time_run_pe(filename, rows):
    cmd = ["../build/run_pe",
           filename,
           "../resources/csv/out/nclusters" + str(rows) +".csv",
           "0", "4"]

    start_time = time.time()
    subprocess.run(cmd, stdout=True)
    end_time = time.time()
    return(end_time - start_time)


def get_dimension(filename):
    rows_str = ""
    cols_str = ""
    l = 0
    for c in filename[8:]:
        l+= 1
        if c == "_": break
        rows_str += c
    for c in filename[8+l:]:
        if c == ".": break
        cols_str += c
    return (int(rows_str), int(cols_str))


def measure_times_col_constant():
    times = []
    for filename in os.listdir(CLUSTER_DIR):
        complete_filename = CLUSTER_DIR + "/" + filename
        if filename.endswith("_10.csv"):
            rows, cols = get_dimension(filename)
            time =  measure_time_run_pe(complete_filename, rows)
            times.append((rows, time))

    times = sorted(times, key=lambda t: t[0])
    return times


def measure_times_row_constant():
    times = []
    for filename in os.listdir(CLUSTER_DIR):
        complete_filename = CLUSTER_DIR + "/" + filename
        if "_800_" in filename:
            rows, cols = get_dimension(filename)
            time =  measure_time_run_pe(complete_filename, rows)
            times.append((cols, time))

    times = sorted(times, key=lambda t: t[0])
    return times


def plot_times(times):
    fig = plt.figure(figsize=(21, 7))
    ax1 = fig.add_subplot()
    # ax1.set_title("posterior loss  with " + LOSS_TYPE + " Loss")
    ax1.plot([times[i][0] for i in range(len(times))],
             [times[i][1] for i in range(len(times))])
    fig.show()
    fig.savefig(PLOT_PATH, format='png')


def main():
    times = measure_times_col_constant()
    #times = measure_times_row_constant()
    plot_times(times)


if __name__ == "__main__":
    # execute only if run as a script
    main()

