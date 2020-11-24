#!/usr/bin/env bash

# build/run \
#   algo_type rng_seed init_num_cl maxiter burnin \
#   hier_type hier_args \
#   mix_type mix_args \
#   datafile gridfile \
#   densfile massfile \
#   nclufile clusfile

if [ "$1" == "uni" ]; then
  build/run \
    N2 20201103 0 1000 100 \
    NNIG resources/nnig_ngg_prior.asciipb \
    DP resources/dp_gamma_prior.asciipb \
    resources/data_uni.csv resources/grid_uni.csv \
    resources/dens_uni.csv resources/mass_uni.csv \
    resources/nclu_uni.csv resources/clus_uni.csv

elif [ "$1" == "multi" ]; then
  build/run \
    N2 20201103 0 1000 100 \
    NNW resources/nnw_ngiw_prior.asciipb \
    DP resources/dp_gamma_prior.asciipb \
    resources/data_multi.csv resources/grid_multi.csv \
    resources/dens_multi.csv resources/mass_multi.csv \
    resources/nclu_multi.csv resources/clus_multi.csv

else
  echo "Syntax: bash/run_test.sh uni (or multi)"
fi
