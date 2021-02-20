#!/usr/bin/env bash

# build/run \
#   algo_type rng_seed init_num_cl maxiter burnin \
#   hier_type hier_args \
#   mix_type mix_args \
#   collname \
#   datafile \
#   gridfile \
#   densfile \
#   massfile \
#   nclufile \
#   clusfile \
#   [hier_cov_file] \
#   [hier_grid_cov_file] \
#   [mix_cov_file] \
#   [mix_grid_cov_file]

if (( $# == 1 )); then
  algo='Neal2'
else
  algo="$2"
fi

if [ "$1" == 'uni' ]; then
  build/run \
    $algo 20201124 0 1000 100 \
    NNIG resources/asciipb/nnig_ngg_prior.asciipb \
    DP   resources/asciipb/dp_gamma_prior.asciipb \
    "" \
    resources/csv/in/data_uni.csv \
    resources/csv/in/grid_uni.csv \
    resources/csv/out/uni_${algo}_dens.csv \
    resources/csv/out/uni_${algo}_mass.csv \
    resources/csv/out/uni_${algo}_nclu.csv \
    resources/csv/out/uni_${algo}_clus.csv
elif [ "$1" == 'multi' ]; then
  build/run \
    $algo 20201124 0 1000 100 \
    NNW resources/asciipb/nnw_ngiw_prior.asciipb \
    DP  resources/asciipb/dp_gamma_prior.asciipb \
    "" \
    resources/csv/in/data_multi.csv \
    resources/csv/in/grid_multi.csv \
    resources/csv/out/multi_${algo}_dens.csv \
    resources/csv/out/multi_${algo}_mass.csv \
    resources/csv/out/multi_${algo}_nclu.csv \
    resources/csv/out/multi_${algo}_clus.csv
elif [ "$1" == 'lru' ]; then
  build/run \
    $algo 20201124 0 1000 100 \
    LinRegUni resources/asciipb/lin_reg_uni_fixed.asciipb \
    DP        resources/asciipb/dp_gamma_prior.asciipb \
    "" \
    resources/csv/in/data_lru.csv \
    resources/csv/in/covs_grid_lru.csv \
    resources/csv/out/lru_${algo}_dens.csv \
    resources/csv/out/lru_${algo}_mass.csv \
    resources/csv/out/lru_${algo}_nclu.csv \
    resources/csv/out/lru_${algo}_clus.csv \
    resources/csv/in/covs_lru.csv \
    resources/csv/in/covs_grid_lru.csv
else
  echo 'Syntax: bash/run_test.sh followed by one of: uni, multi, lru'
fi
