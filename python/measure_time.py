import matplotlib.pyplot as plt
import os
import subprocess
import time

# paths needed
CLUSTER_DIR = "../resources/csv/in/clusters"
PLOT_PATH = "../resources/csv/plot/times_line_constant.png"


def measure_time_run_pe(filename, rows, loss):
    cmd = ["../build/run_pe",
           filename,
           "../resources/csv/out/nclusters" + str(rows) +".csv",
           str(loss), "4"]

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
    time_binder = []
    time_vi = []
    time_vin = []
    for filename in os.listdir(CLUSTER_DIR):
        complete_filename = CLUSTER_DIR + "/" + filename
        if filename.endswith("_10.csv"):
            rows, cols = get_dimension(filename)
            time1 =  measure_time_run_pe(complete_filename, rows, 0)
            time2 =  measure_time_run_pe(complete_filename, rows, 1)
            time3 =  measure_time_run_pe(complete_filename, rows, 2)
            time_binder.append((rows, time1))
            time_vi.append((rows, time2))
            time_vin.append((rows, time3))

    time_binder = sorted(time_binder, key=lambda t: t[0])
    time_vi = sorted(time_vi, key=lambda t: t[0])
    time_vin = sorted(time_vin, key=lambda t: t[0])


    print("col constant")
    print(time_binder)
    print(time_vi)
    print(time_vin)
    return (time_binder, time_vi, time_vin)




# [(100, 0.6975607872009277), (200, 1.3121731281280518), (300, 1.081101417541504), (400, 1.7171852588653564), (500, 1.6293609142303467), (600, 2.343804359436035), (700, 2.24833607673645), (800, 3.2639622688293457), (900, 3.7223806381225586), (1000, 4.458684682846069), (1100, 2.961268186569214), (1200, 4.419412851333618), (1500, 4.943319320678711), (1800, 7.755052328109741), (2000, 9.641566038131714), (2500, 7.473871231079102), (3000, 9.18504786491394), (3500, 9.995590686798096), (4000, 21.148584127426147), (4500, 16.702173709869385), (5000, 16.073558807373047)]
# [(100, 5.167551517486572), (200, 5.8958563804626465), (300, 5.146724700927734), (400, 4.618350505828857), (500, 5.6942713260650635), (600, 9.82329535484314), (700, 11.33936595916748), (800, 9.54245138168335), (900, 22.0994074344635), (1000, 23.36195158958435), (1100, 12.17194676399231), (1200, 27.603338479995728), (1500, 30.22098422050476), (1800, 20.14135241508484), (2000, 38.5941424369812), (2500, 57.341344118118286), (3000, 53.71455502510071), (3500, 86.12110829353333), (4000, 91.77992677688599), (4500, 68.62427473068237), (5000, 105.48212313652039)]
# [(100, 4.08366847038269), (200, 5.630253791809082), (300, 5.961718559265137), (400, 19.882601499557495), (500, 10.173685789108276), (600, 23.963545083999634), (700, 15.452309608459473), (800, 19.222697019577026), (900, 38.62432885169983), (1000, 28.028536558151245), (1100, 32.66293025016785), (1200, 27.820096492767334), (1500, 36.749364137649536), (1800, 39.9458270072937), (2000, 70.92197823524475), (2500, 73.83657646179199), (3000, 66.14072918891907), (3500, 219.86626958847046), (4000, 110.73812556266785), (4500, 129.17683863639832), (5000, 179.62309646606445)]

def measure_times_row_constant():
    time_binder = []
    time_vi = []
    time_vin = []
    for filename in os.listdir(CLUSTER_DIR):
        complete_filename = CLUSTER_DIR + "/" + filename
        if "_800_" in filename:
            rows, cols = get_dimension(filename)
            time1 =  measure_time_run_pe(complete_filename, rows, 0)
            time2 =  measure_time_run_pe(complete_filename, rows, 1)
            time3 =  measure_time_run_pe(complete_filename, rows, 2)
            time_binder.append((cols, time1))
            time_vi.append((cols, time2))
            time_vin.append((cols, time3))

    time_binder = sorted(time_binder, key=lambda t: t[0])
    time_vi = sorted(time_vi, key=lambda t: t[0])
    time_vin = sorted(time_vin, key=lambda t: t[0])

    print("line constant")
    print(time_binder)
    print(time_vi)
    print(time_vin)
    return (time_binder, time_vi, time_vin)


def plot_times(time_binder, time_vi, time_vin):
    fig = plt.figure(figsize=(21, 7))
    ax1 = fig.add_subplot()
    # ax1.set_title("posterior loss  with " + LOSS_TYPE + " Loss")
    ax1.plot([time_binder[i][0] for i in range(len(time_binder))],
             [time_binder[i][1] for i in range(len(time_binder))], 'r',  label='Binder Loss')
    ax1.plot([time_vi[i][0] for i in range(len(time_vi))],
             [time_vi[i][1] for i in range(len(time_vi))], 'g',  label='VI Loss')
    ax1.plot([time_vin[i][0] for i in range(len(time_vin))],
             [time_vin[i][1] for i in range(len(time_vin))], 'b',  label='VIN loss')


    ax1.legend()
    ax1.plot()
    plt.xlabel('length of dataset')
    plt.ylabel('time (s)')
    plt.title('Computation time for each loss function')
    fig.show()
    fig.savefig(PLOT_PATH, format='png')


def main():
    # time_binder, time_vi, time_vin = measure_times_col_constant()
    time_binder, time_vi, time_vin = measure_times_row_constant()
    plot_times(time_binder, time_vi, time_vin)

if __name__ == "__main__":
    # execute only if run as a script
    main()

