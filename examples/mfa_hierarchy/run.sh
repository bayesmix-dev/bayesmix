#!/usr/bin/env bash

# build/run \
#   algorithm_settings_file \
#   hierarchy_type  hierarchy_prior_file \
#   mixing_type  mixing_prior_file \
#   collector_name \
#   data_file \
#   grid_file \
#   density_output_file \
#   numclust_output_file \
#   clustering_output_file \
#   [hierarchy_covariates_file] \
#   [hierarchy_grid_covariates_file] \
#   [mixing_covariates_file] \
#   [mixing_grid_covariates_file]

#build/run \
#  examples/examples_mfa_hierarchy/in/algo.asciipb \
#  MFA examples/examples_mfa_hierarchy/in/nnig_ngg.asciipb \
#  DP   examples/examples_mfa_hierarchy/in/dp_gamma.asciipb \
#  examples/examples_mfa_hierarchy/out/chains.recordio \
#  examples/examples_mfa_hierarchy/in/data.csv \
#  examples/examples_mfa_hierarchy/in/grid.csv \
#  examples/examples_mfa_hierarchy/out/density_file.csv \
#  examples/examples_mfa_hierarchy/out/numclust.csv \
#  examples/examples_mfa_hierarchy/out/clustering.csv

  build/run_mcmc \
  --algo-params-file examples/mfa_hierarchy/in/algo.asciipb \
  --hier-type MFA --hier-args examples/mfa_hierarchy/in/nnig_ngg.asciipb \
  --mix-type DP --mix-args examples/mfa_hierarchy/in/dp_gamma.asciipb \
  --coll-name examples/mfa_hierarchy/out/chains.recordio \
  --data-file examples/mfa_hierarchy/in/data.csv \
  --grid-file examples/mfa_hierarchy/in/grid.csv \
  --dens-file examples/mfa_hierarchy/out/density_file.csv \
  --n-cl-file examples/mfa_hierarchy/out/numclust.csv \
  --clus-file examples/mfa_hierarchy/out/clustering.csv \
  #--best-clus-file resources/tutorial/out/best_clustering.csv
