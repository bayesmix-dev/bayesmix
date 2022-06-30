import os
import subprocess

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from collections import defaultdict
from contextlib import redirect_stdout

from bayesmixpy import build_bayesmix, run_mcmc

os.environ["BAYESMIX_EXE"] = "../build/run_mcmc"

OUTPUT_PATH = "reproducibility"

ALGORITHMS = "Neal2 Neal3 Neal8 SplitMerge".split()

ALGO_SETTINGS = """
        algo_id: "{}"
        rng_seed: 20201124
        iterations: 5000
        burnin: 1000
        init_num_clusters: 3
        neal8_n_aux: 3
        splitmerge_n_restr_gs_updates: 5
        splitmerge_n_mh_updates: 1
        splitmerge_n_full_gs_updates: 1
        """

PY_PRIOR = """
        fixed_values {
            strength: 1.0
            discount: 0.1
        }"""


GALAXY_GO = """
        ngg_prior {
        mean_prior {
            mean: 25.0
            var: 4.0
        }
        var_scaling_prior {
            shape: 0.4
            rate: 0.2
        }
        shape: 4.0
        scale_prior {
            shape: 4.0
            rate: 2.0
        }
        }
        """

FAITHFUL_G0 = """
    ngiw_prior {
        mean_prior {
            mean {
            size: 2
            data: [3.0, 3.0]
            }
            var {
            rows: 2
            cols: 2
            data: [0.25, 0.0, 0.0, 0.25]
            }
        }
        var_scaling_prior {
            shape: 0.4
            rate: 0.2
        }
        deg_free: 4.0
        scale_prior {
            deg_free: 4.0
            scale {
            rows: 2
            cols: 2
            data: [4.0, 0.0, 0.0, 4.0]
            }
        }
    }
    """

HIGHDIM_G0 = """
   ngiw_prior {
  mean_prior {
    mean {
      size: 4
      data: [0.0, 0.0, 0.0, 0.0]
    }
    var {
      rows: 4
      cols: 4
      data: [0.1, 0.0, 0.0, 0.0,
             0.0, 0.1, 0.0, 0.0,
             0.0, 0.0, 0.1, 0.0,
             0.0, 0.0, 0.0, 0.1]
    }
  }
  var_scaling_prior {
    shape: 0.2
    rate: 2.0
  }
  deg_free: 10.0
  scale_prior {
    deg_free: 10.0
    scale {
      rows: 4
      cols: 4
      data: [0.1, 0.0, 0.0, 0.0,
             0.0, 0.1, 0.0, 0.0,
             0.0, 0.0, 0.1, 0.0,
             0.0, 0.0, 0.0, 0.1]
    }
  }
}
"""

def generate_highdim_data(outfile):
    rng_seed = 20201124
    dim = 4
    points_per_clust = 5000

    # Initialize mean and covariance matrix
    mean = np.array(dim*[2])
    cov = np.eye(dim)

    # Generate dataset for two clusters
    rng = np.random.default_rng(rng_seed)
    samples1 = rng.multivariate_normal(mean=+mean, cov=cov, size=points_per_clust)
    samples2 = rng.multivariate_normal(mean=-mean, cov=cov, size=points_per_clust)
    samples = np.vstack((samples1, samples2))

    # Save joined datasets to file
    np.savetxt(outfile, samples, delimiter=',', fmt="%.5f")


def run_bayesmix(log_fold, dataset, name, g0_params):
    out_dens = defaultdict(dict)
    out_clus = defaultdict(dict)
    g0_name = "NNIG" if name == "galaxy" else "NNW"
    for algo in ALGORITHMS:
        log_file = os.path.join(log_fold, 'bayesmix_{0}_{1}.log'.format(name, algo))
        with open(log_file, 'w') as f:
            with redirect_stdout(f):
                out = run_mcmc(g0_name, "PY", dataset, g0_params, PY_PRIOR,
                            ALGO_SETTINGS.format(algo), dataset,
                            return_num_clusters=True,  # out [1]
                            return_clusters=False, return_best_clus=False)
        out_dens[name][algo] = out[0]
        out_clus[name][algo] = out[1]
    return out_dens, out_clus



if __name__ == "__main__":
    log_fold = os.path.join(OUTPUT_PATH, "log")
    csv_fold = os.path.join(OUTPUT_PATH, "csv")
    png_fold = os.path.join(OUTPUT_PATH, "png")

    for fold in [log_fold, csv_fold, png_fold]:
        os.makedirs(fold, exist_ok=True)

    build_bayesmix(4)

    highdim_data_file = os.path.join(csv_fold, "highdim_data.csv")
    generate_highdim_data(highdim_data_file)

    datasets = {}
    datasets["galaxy"] = np.loadtxt('../resources/datasets/galaxy.csv', delimiter=',')
    datasets["faithful"] = np.loadtxt('../resources/datasets/faithful.csv', delimiter=',')
    datasets["highdim"] = np.loadtxt(highdim_data_file, delimiter=",")

    g0_params = {}
    g0_params["galaxy"] = GALAXY_GO
    g0_params["faithful"] = FAITHFUL_G0
    g0_params["highdim"] = HIGHDIM_G0

    bayesmix_densities = {}
    bayesmix_num_clust = {}

    for name in datasets.keys():
        print("RUNNING BAYESMIX FOR " + name)
        curr_dens, curr_clust = run_bayesmix(
            log_fold, datasets[name], name, g0_params[name])
        bayesmix_densities.update(curr_dens)
        bayesmix_num_clust.update(curr_clust)

    ## RUN BNPMIX VIA SUBPROCESS
    bnpmix_algos = ('MAR', 'ICS')
    subprocess.call("Rscript run_bnpmix.R".split(), cwd="reproducibility")

    # COMPUTE ESS and TIMES
    datasets = ["galaxy", "faithful", "highdim"]
    ESS = pd.DataFrame(columns=datasets)
    times = pd.DataFrame(columns=datasets)
    for data in datasets:
        for algo in bnpmix_algos:
            csv_file = os.path.join(
                csv_fold, 'bnpmix_{}_nclu_{}.csv'.format(data, algo))
            n_clust = np.genfromtxt(csv_file)
            ESS.at['bnpmix_'+algo, data] = az.ess(n_clust)

            log_file = os.path.join(
                log_fold, 'bnpmix_{}_{}.log'.format(data, algo))
            with open(log_file, 'r') as f:
                for line in f:
                    if "Estimation done in " in line:
                        time = line.split()[3]
                        times.at['bnpmix_'+algo, data] = float(time)
                        break

        for algo in ALGORITHMS:
            ESS.at['bayesmix_'+algo, data] = az.ess(bayesmix_num_clust[data][algo])
            log_file = os.path.join(
                log_fold, 'bayesmix_{}_{}.log'.format(data, algo))
            with open(log_file, 'r') as f:
                for line in f:
                    if "100%" in line and "Done" in line:
                        time = line.split()[2].rstrip("s")
                        times.at['bayesmix_'+algo, data] = float(time)
                        break

    ratios = pd.DataFrame()
    for col in ESS.columns:
        ratios[col] = ESS[col] / times[col]

    # DISPLAY METRICS TABLE
    metric_names = 'ESS times ratios'.split()
    for data in datasets:
        df_all = pd.DataFrame(index=ESS.index, columns=metric_names)
        for metric in metric_names:
            df = globals()[metric]
            df_all[metric] = np.round(df[data].astype(float), 3)
        print(data, ":\n", df_all, "\n", sep="")

    # AUTOCORRELATION PLOTS
    size = 20
    for data in datasets:
        for algo in bnpmix_algos:
            csv_file = os.path.join(
                csv_fold, 'bnpmix_{}_nclu_{}.csv'.format(data, algo))
            n_clust = np.genfromtxt(csv_file)
            ax = az.plot_autocorr(n_clust)
            ax.set_xlabel("lag", size=size)
            ax.set_ylabel("autocorrelation", size=size)
            ax.set_title("BNPmix {} {}".format(data,algo), size=size)
            plt.savefig(
                os.path.join(png_fold, 'bnpmix_{}_{}.png'.format(data, algo)),
                dpi=300, bbox_inches='tight')

        for algo in ALGORITHMS:
            ax = az.plot_autocorr(bayesmix_num_clust[data][algo])
            ax.set_xlabel("lag", size=size)
            ax.set_ylabel("autocorrelation", size=size)
            ax.set_title("bayesmix {} {}".format(data,algo), size=size)
            plt.savefig(
                os.path.join(png_fold, 'bayesmix_{}_{}.png'.format(data, algo)),
                dpi=300, bbox_inches='tight')
