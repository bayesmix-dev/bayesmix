# Tests for run_mcmc function

DP_PARAMS =
"
fixed_value {
    totalmass: 1.0
}
"

GO_PARAMS =
"
fixed_values {
    mean: 0.0
    var_scaling: 0.1
    shape: 2.0
    scale: 2.0
}
"

ALGO_PARAMS =
"
algo_id: 'Neal2'
rng_seed: 20201124
iterations: 10
burnin: 5
init_num_clusters: 3
"

get_data = function() {
  set.seed(0)
  return(rnorm(10))
}

get_grid = function(){
  return(seq(-5, 5, length.out = 10))
}

test_run_mcmc = function() {

  data = get_data()
  grid = get_grid()
  readRenviron(system.file(".Renviron", package = "bayesmixr"))

  out = bayesmixr::run_mcmc("NNIG", "DP", data, GO_PARAMS, DP_PARAMS, ALGO_PARAMS, grid,
                            return_clusters=FALSE, return_num_clusters=FALSE, return_best_clus=FALSE)

  stopifnot(dim(out$eval_dens)[1] == 5)
  stopifnot(dim(out$eval_dens)[2] == length(grid))
  stopifnot(is.null(out$n_clus))
  stopifnot(is.null(out$clus))
  stopifnot(is.null(out$best_clus))

  out = bayesmixr::run_mcmc("NNIG", "DP", data, GO_PARAMS, DP_PARAMS, ALGO_PARAMS, NULL,
                            return_clusters=FALSE, return_num_clusters=TRUE, return_best_clus=FALSE)

  stopifnot(is.null(out$eval_dens))
  stopifnot(!is.null(out$n_clus))
  stopifnot(length(out$n_clus) == 5)
  stopifnot(is.null(out$clus))
  stopifnot(is.null(out$best_clus))

  out = bayesmixr::run_mcmc("NNIG", "DP", data, GO_PARAMS, DP_PARAMS, ALGO_PARAMS, NULL,
                            return_clusters=TRUE, return_num_clusters=FALSE, return_best_clus=FALSE)

  stopifnot(is.null(out$eval_dens))
  stopifnot(is.null(out$n_clus))
  stopifnot(!is.null(out$clus))
  stopifnot(dim(out$clus)[1] == 5)
  stopifnot(dim(out$clus)[2] == length(data))
  stopifnot(is.null(out$best_clus))


  out = bayesmixr::run_mcmc("NNIG", "DP", data, GO_PARAMS, DP_PARAMS, ALGO_PARAMS, NULL,
                            return_clusters=FALSE, return_num_clusters=FALSE, return_best_clus=TRUE)

  stopifnot(is.null(out$eval_dens))
  stopifnot(is.null(out$n_clus))
  stopifnot(is.null(out$clus))
  stopifnot(!is.null(out$best_clus))
  stopifnot(length(out$best_clus) == length(data))
}
