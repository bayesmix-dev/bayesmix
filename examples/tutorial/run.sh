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

build/run \
  --algo_params_file resources/tutorial/algo.asciipb \
  --hier_type NNIG --hier_args resources/tutorial/nnig_ngg.asciipb \
  --mix_type DP --mix_args resources/tutorial/dp_gamma.asciipb \
  --collname resources/tutorial/out/chains.recordio \
  --datafile resources/tutorial/data.csv \
  --gridfile resources/tutorial/grid.csv \
  --densfile resources/tutorial/out/density_file.csv \
  --nclufile resources/tutorial/out/numclust.csv \
  --clusfile resources/tutorial/out/clustering.csv
