# BayesMixR: an R interface to BayesMix

## Installation

The simplest way to install `bayesmixr` on all platforms is via [`devtools`](https://cran.r-project.org/web/packages/devtools/index.html) package in `R`. After you have cloned the `bayesmix` GitHub directory, open `R`, navigate to the `R/` sub-folder and install `bayesmixr` via:

```r
# Install devtools in case is not present
install.packages("devtools")

# Locally install bayesmixr and clean files created at installation time
devtools::install("bayesmixr/", quick = TRUE, args = "--clean")
```

## Usage

`bayesmixr` provides two main functions: `build_bayesmix` and `run_mcmc`. The first one installs `bayesmix` and its executables for you, while the second one calls the executable that runs the MCMC sampler from `R`.

### Building bayesmix

To build `bayesmix`, in a R/Rstudio session or script write

```r
# load library
library("bayesmixr")

# Set number of processors for parallel build (it defaults to half of your cores)
n_proc = 4

# Build bayesmix on your system
build_bayesmix(n_proc)
```

This will print out the full installation log.

### Running bayesmix

To `run_mcmc`, you must define the model and the algorithm in some configuration files or text strings. See the documentation for more details.

For instance, to fit a Dirichlet Process Mixture on univariate data using a Normal-NormalInverseGamma hierarchy using Neal's Algorithm 3, we use the following

```r
out = run_mcmc("NNIG", "DP", data, nnig_params, dp_params, algo_params, dens_grid)
```

where `data` is a numeric vector of data points, `dens_grid` is a numeric vector of points where to evaluate the density, and `nnig_params`, `dp_params` and `algo_params` are defined as follows.

```r
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

This specifies that the base (centering) measure is a Normal-InverseGamma with parameters $(\mu_0, \lambda_0, a_0, b_0)$. Moreover, $\mu_0 \sim \mathcal{N}(5.5, 2.25)$, $\lambda_0 \sim \mathcal{G}(0.2, 0.6)$, $a_0 = 1.5$ and $b_0 \sim \mathcal{G}(4.0, 2.0)$. See the messages `NNIGPrior` and `NNIGPrior::NGGPrior` in the file [hierarchy_prior.proto](https://github.com/bayesmix-dev/bayesmix/blob/master/src/proto/hierarchy_prior.proto) for further reference.

```r
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

This specifies that the concentration parameter of the DP has an hyperprior which is a Gamma distribution with parameters (4, 2). Finally, we specify the parameters of the algorithm as follows:

```r
algo_params =
  "
  algo_id: 'Neal3'
  rng_seed: 20201124
  iterations: 2000
  burnin: 1000
  init_num_clusters: 3
  "
```

See the notebook in `notebooks/gaussian_mix_uni.Rmd` for a concrete usage example
