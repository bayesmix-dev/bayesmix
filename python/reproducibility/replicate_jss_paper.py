import os
import shutil
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
# ALGORITHMS = ["Neal3"]

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

DP_PRIOR = """
    fixed_value {
        totalmass: 1.0
}
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

EXAMPLE_G0 = """
fixed_values {
    mean: 0.0
    var_scaling: 0.1
    shape: 2.0
    scale: 2.0
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


def run_bayesmix(log_fold, dataset, name, g0_params, univariate: bool):
    out_dens = defaultdict(dict)
    out_clus = defaultdict(dict)
    g0_name = "NNIG" if univariate else "NNW"
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
    build_fold = os.path.join(OUTPUT_PATH, os.pardir, os.pardir, "build")

    shutil.rmtree(build_fold, ignore_errors=True)

    for fold in (log_fold, csv_fold, png_fold, build_fold):
        os.makedirs(fold, exist_ok=True)

    subprocess.call("cmake .. -DDISABLE_BENCHMARKS=ON".split(), cwd=build_fold)
    subprocess.call("make plot_mcmc -j4".split(), cwd=build_fold)

    build_bayesmix(4)

    ###########################
    ## COMMAND LINE EXAMPLES ##
    ###########################

    run_cmd = """build/run_mcmc
            --algo-params-file resources/tutorial/algo.asciipb
            --hier-type NNIG --hier-args resources/tutorial/nnig_ngg.asciipb
            --mix-type DP --mix-args resources/tutorial/dp_gamma.asciipb
            --coll-name python/{0}/chains.recordio
            --data-file resources/tutorial/data.csv
            --grid-file resources/tutorial/grid.csv
            --dens-file python/{1}/cmdline_density.csv
            --n-cl-file python/{1}/cmdline_numclust.csv
            --clus-file python/{1}/cmdline_clustering.csv
            --best-clus-file python/{1}/cmdline_best_clustering.csv
    """.format(log_fold, csv_fold)
    subprocess.call(run_cmd.split(), cwd="../")

    plt_cmd = """build/plot_mcmc
        --grid-file resources/tutorial/grid.csv
        --dens-file python/{0}/cmdline_density.csv
        --dens-plot python/{1}/cmdline_density.png
        --n-cl-file python/{0}/cmdline_numclust.csv
        --n-cl-trace-plot python/{1}/cmdline_traceplot.png
        --n-cl-bar-plot  python/{1}/cmdline_nclus_barplot.png
    """.format(csv_fold, png_fold)
    subprocess.call(plt_cmd.split(), cwd="../")

    ###############################
    ## PYTHON UNIVARIATE EXAMPLE ##
    ###############################
    data = np.concatenate([
        np.random.normal(loc=3, scale=1, size=100),
        np.random.normal(loc=-3, scale=1, size=100),
    ])

    dp_params = """
    fixed_value {
        totalmass: 1.0
    }
    """

    g0_params = """
    fixed_values {
        mean: 0.0
        var_scaling: 0.1
        shape: 2.0
        scale: 2.0
    }
    """

    algo_params = """
        algo_id: "Neal2"
        rng_seed: 20201124
        iterations: 2000
        burnin: 1000
        init_num_clusters: 3
    """

    dens_grid = np.linspace(-6, 6, 1000)

    log_dens, numcluschain, cluschain, bestclus = run_mcmc(
        "NNIG", "DP", data, g0_params, dp_params, algo_params,
        dens_grid=dens_grid, return_clusters=True, return_num_clusters=True,
        return_best_clus=True)

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    axes[0].hist(data, alpha=0.2, density=True)
    for c in np.unique(bestclus):
        data_in_clus = data[bestclus == c]
        axes[0].scatter(data_in_clus, np.zeros_like(data_in_clus) + 0.01,
                        label="Cluster {0}".format(int(c) + 1))
    axes[0].plot(dens_grid, np.exp(np.mean(log_dens, axis=0)), color="red", lw=3)
    axes[0].legend(fontsize=12, ncol=2, loc=1)
    axes[0].set_ylim(0, 0.3)


    x, y = np.unique(numcluschain, return_counts=True)
    axes[1].bar(x, y / y.sum())
    axes[1].set_xticks(x)

    axes[2].vlines(np.arange(len(numcluschain)), numcluschain-0.3, numcluschain+0.3)
    plt.savefig(os.path.join(png_fold, 'bayesmix_example_univariate.png'),
                dpi=300, bbox_inches='tight')


    ##############################
    ## PYTHON BIVARIATE EXAMPLE ##
    ##############################

    g0_params = """
    fixed_values {
        mean {
            size: 2
            data: [3.484, 3.487]
        }
        var_scaling: 0.01
        deg_free: 5
        scale {
            rows: 2
            cols: 2
            data: [1.0, 0.0, 0.0, 1.0]
            rowmajor: false
        }
    }
    """

    data = np.loadtxt('../resources/datasets/faithful.csv', delimiter=',')
    xgrid = np.linspace(0, 6, 50)
    xgrid, ygrid = np.meshgrid(xgrid, xgrid)
    dens_grid = np.hstack([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)])

    log_dens, numcluschain, _, best_clus_dp = run_mcmc(
        "NNW", "DP", data, g0_params, dp_params, algo_params,
        dens_grid, return_clusters=False, return_num_clusters=True,
        return_best_clus=True)

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    mean_dens_dp = np.mean(log_dens, axis=0)

    axes[0].contour(xgrid, ygrid, mean_dens_dp.reshape(xgrid.shape))
    for c in np.unique(best_clus_dp):
        currdata = data[best_clus_dp == c, :]
        axes[0].scatter(currdata[:, 0], currdata[:, 1])

    x, y = np.unique(numcluschain, return_counts=True)
    axes[1].bar(x, y / y.sum())
    axes[1].set_xticks(x)

    axes[2].vlines(np.arange(len(numcluschain)), numcluschain-0.3, numcluschain+0.3)
    plt.savefig(os.path.join(png_fold, 'bayesmix_example_bivariate.png'),
                dpi=300, bbox_inches='tight')


    ############################
    ## COMPARISON WITH BNPMIX ##
    ############################

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
        univariate = True if name == "galaxy" else False
        curr_dens, curr_clust = run_bayesmix(
            log_fold, datasets[name], name, g0_params[name], univariate)
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
