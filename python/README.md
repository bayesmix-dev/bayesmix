# BayesMixPy: a Python interface to BayesMix

## Installation

After you have cloned the bayesmix github directory, navigate to the Python subfolder and
install bayesmixpy using pip

```
cd python
pip3 install -e .
```

## Usage

`bayesmixpy` provides two functions: `build_bayesmix` and `run_mcmc`. The first one
installs `bayesmix` and its executables for you, while the second one calls the
executable that runs the MCMC sampler from Python.

### Building bayesmix

To build `bayesmix`, in a Python shell or a notebook write

```
from bayesmixpy import build_bayesmix

n_proc = 4 # number of processors for building in parallel
build_bayesmix(n_proc)
```

this will print out the installation log and, if the installation was successful, the following message

```
Bayesmix executable is in '<BAYESMIX_HOME_REPO>/build',
export the environment variable BAYESMIX_EXE=<BAYESMIX_HOME_REPO>build/run
```

Hence, for running the MCMC chain you should export the `BAYESMIX_EXE` environment variable. This can be done once and for all by copying

```
BAYESMIX_EXE=<BAYESMIX_HOME_REPO>build/run
```

in your .bashrc file (or .zshrc if you are a MacOs user), or every time you use bayesmixpy, you can add the following lines on top of your Python script/notebook

```
import os
os.environ["BAYESMIX_EXE"] = <BAYESMIX_HOME_REPO>build/run

from bayesmixpy import run_mcmc
....
```

### Running bayesmix

To `run_mcmc` must define the model and the algorithm in some configuration files or
text strings. See the documentation for more details.

For instance, to fit a Dirichlet Process Mixture on univariate data using a Normal-Normal-InverseGamma hierarchy using Neal's Algorithm 3, we use the following

```
eval_dens, n_clus_chain, best_clus = run_mcmc(
    "NNIG",
    "DP",
    data,
    nnig_params,
    dp_params,
    algo_params,
    dens_grid)
```

where `data` is a np.array of data points, `dens_grid` is a np.array of points where to evaluate the density, and `nnig_params`, `dp_params` and `algo_params` are defined as follows.

```
nnig_params="""
ngg_prior {
  mean_prior {
    mean: 5.5
    var: 2.25
  }
  var_scaling_prior {
    shape: 0.2
    rate: 0.6
  }
  shape: 1.5
  scale_prior {
    shape: 4.0
    rate: 2.0
  }
}
"""
```

This specifies that the base (centering) measure is a Normal-InverseGamma with parameters (mu0, lam0, a0, b0). Further, mu0 ~ N(5.5, 2.25) lam0 ~ Gamma(0.2, 0.6) a0 = 1.5, b0 ~ Gamma(4.0, 2.0) [See the messages NNIGPrior and NNIGPrior::NGGPrior in the file hierarchy_prior.proto for further reference].

```
dp_params = """
gamma_prior {
  totalmass_prior {
    shape: 4.0
    rate: 2.0
  }
}
"""
```

This specifies that the concentration parameter of the DP has an hyperprior which is a Gamma distribution with parameters (4, 2).

```
algo_params = """
algo_id: "Neal3"
rng_seed: 20201124
iterations: 2000
burnin: 1000
init_num_clusters: 3
"""
```

Se the notebook in `notebooks/gaussian_mix_uni.ipynb` for a concrete usage example
