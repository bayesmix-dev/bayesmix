# BayesMixR: an R interface to BayesMix

## Installation

After you have cloned the bayesmix github directory, navigate to the R subfolder and
install bayesmixr using R via

```shell
cd R
R CMD INSTALL bayesmixr
```

## Usage

`bayesmixr` provides two functions: `build_bayesmix` and `run_mcmc`. The first one
installs `bayesmix` and its executables for you, while the second one calls the
executable that runs the MCMC sampler from Python.

### Building bayesmix

To build `bayesmix`, in a R/Rstudio session or script write

```R
library("bayesmixr")

n_proc = 4 # number of processors for building in parallel (defaults to half of your cores)
build_bayesmix(n_proc)
```

this will print out the installation log and, if the installation was successful, it will automatically export the `BAYESMIX_EXE` environment variable, which is required by the `run_mcmc` function to proper call `bayesmix` library.

### Running bayesmix

To `run_mcmc`, you must define the model and the algorithm in some configuration files or
text strings. See the documentation for more details.

For instance, to fit a Dirichlet Process Mixture on univariate data using a Normal-Normal-InverseGamma hierarchy using Neal's Algorithm 3, we use the following

```R
out = run_mcmc("NNIG", "DP", data, nnig_params, dp_params, algo_params, dens_grid)
```

where `data` is a numeric vector of data points, `dens_grid` is a numeric vector of points where to evaluate the density, and `nnig_params`, `dp_params` and `algo_params` are defined as follows.

```R
nnig_params =
"
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
"
```

This specifies that the base (centering) measure is a Normal-InverseGamma with parameters (mu0, lam0, a0, b0). Further, mu0 ~ N(5.5, 2.25) lam0 ~ Gamma(0.2, 0.6) a0 = 1.5, b0 ~ Gamma(4.0, 2.0) [See the messages NNIGPrior and NNIGPrior::NGGPrior in the file hierarchy_prior.proto for further reference].

```R
dp_params =
"
gamma_prior {
  totalmass_prior {
    shape: 4.0
    rate: 2.0
  }
}
"
```

This specifies that the concentration parameter of the DP has an hyperprior which is a Gamma distribution with parameters (4, 2).

```R
algo_params = "
algo_id: "Neal3"
rng_seed: 20201124
iterations: 2000
burnin: 1000
init_num_clusters: 3
"
```

See the notebook in `notebooks/gaussian_mix_uni.Rmd` for a concrete usage example
