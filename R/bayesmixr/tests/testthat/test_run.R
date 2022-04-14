test_that("run_mcmc() is successful", {

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

  data = rnorm(10)
  grid = seq(-5, 5, length.out = 10)

  # 1st bunch of tests
  cat("\n")
  out = run_mcmc("NNIG", "DP", data, GO_PARAMS, DP_PARAMS, ALGO_PARAMS, grid,
                 return_clusters=FALSE, return_num_clusters=FALSE, return_best_clus=FALSE)
  expect_type(out$eval_dens, "double")
  expect_equal(dim(out$eval_dens)[1], 5)
  expect_equal(dim(out$eval_dens)[2], length(grid))
  expect_null(out$n_clus)
  expect_null(out$clus)
  expect_null(out$best_clus)

  # 2nd bunch of tests
  cat("\n")
  out = run_mcmc("NNIG", "DP", data, GO_PARAMS, DP_PARAMS, ALGO_PARAMS, NULL,
                 return_clusters=FALSE, return_num_clusters=TRUE, return_best_clus=FALSE)
  expect_null(out$eval_dens)
  expect_type(out$n_clus, "integer")
  expect_equal(length(out$n_clus), 5)
  expect_null(out$clus)
  expect_null(out$best_clus)

  # 3rd bunch of tests
  cat("\n")
  out = run_mcmc("NNIG", "DP", data, GO_PARAMS, DP_PARAMS, ALGO_PARAMS, NULL,
                 return_clusters=TRUE, return_num_clusters=FALSE, return_best_clus=FALSE)
  expect_null(out$eval_dens)
  expect_null(out$n_clus)
  expect_type(out$clus, "integer")
  expect_equal(dim(out$clus)[1], 5)
  expect_equal(dim(out$clus)[2], length(data))
  expect_null(out$best_clus)

  # 4th bunch of tests
  cat("\n")
  out = run_mcmc("NNIG", "DP", data, GO_PARAMS, DP_PARAMS, ALGO_PARAMS, NULL,
                 return_clusters=FALSE, return_num_clusters=FALSE, return_best_clus=TRUE)
  expect_null(out$eval_dens)
  expect_null(out$n_clus)
  expect_null(out$clus)
  expect_type(out$best_clus, "integer")
  expect_equal(length(out$best_clus), length(data))
})
