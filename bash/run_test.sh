#!/usr/bin/env bash

# build/run \
#   algo_type rng_seed init_num_cl maxiter burnin \
#   hier_type hier_args \
#   mix_type mix_args \
#   datafile gridfile \
#   densfile massfile \
#   nclufile clusfile \
#   hier_cov_file grid_cov_file

if [ "$1" == "uni" ]; then
  build/run \
    N2 20201124 0 1000 100 \
    NNIG resources/asciipb/nnig_ngg_prior.asciipb \
    DP resources/asciipb/dp_gamma_prior.asciipb \
    resources/csv/in/data_uni.csv resources/csv/in/grid_uni.csv \
    resources/csv/out/un_dens.csv resources/csv/out/un_mass.csv \
    resources/csv/out/un_nclu.csv resources/csv/out/un_clus.csv

elif [ "$1" == "multi" ]; then
  build/run \
    N2 20201124 0 1000 100 \
    NNW resources/asciipb/nnw_ngiw_prior.asciipb \
    DP resources/asciipb/dp_gamma_prior.asciipb \
    resources/csv/in/data_multi.csv resources/csv/in/grid_multi.csv \
    resources/csv/out/mn_dens.csv resources/csv/out/mn_mass.csv \
    resources/csv/out/mn_nclu.csv resources/csv/out/mn_clus.csv

elif [ "$1" == "lru" ]; then
  build/run \
    N2 20201124 0 1000 100 \
#   LinRegUni hier_args \
    DP resources/asciipb/dp_gamma_prior.asciipb \
#   datafile gridfile \
    resources/csv/out/lru_dens.csv resources/csv/out/lru_mass.csv \
    resources/csv/out/lru_nclu.csv resources/csv/out/lru_clus.csv \
#   hier_cov_file grid_cov_file


else
  echo "Syntax: bash/run_test.sh followed by one of: uni, multi, lru"
fi
