#!/usr/bin/env bash

# build/run \
#   algo_params_file \
#   hier_type hier_args \
#   mix_type mix_args \
#   collname \
#   datafile \
#   gridfile \
#   densfile \
#   nclufile \
#   clusfile \
#   [hier_cov_file] \
#   [hier_grid_cov_file] \
#   [mix_cov_file] \
#   [mix_grid_cov_file]

if [ "$1" == 'uni' ]; then
  build/run \
    algo_marg_settings.asciipb \
    NNIG resources/asciipb/nnig_ngg_prior.asciipb \
    DP   resources/asciipb/dp_gamma_prior.asciipb \
    "" \
    resources/csv/in/uni_data.csv \
    resources/csv/in/uni_grid.csv \
    resources/csv/out/uni_dens.csv \
    resources/csv/out/uni_nclu.csv \
    resources/csv/out/uni_clus.csv
elif [ "$1" == 'multi' ]; then
  build/run \
    algo_marg_settings.asciipb \
    NNW resources/asciipb/nnw_ngiw_prior.asciipb \
    DP  resources/asciipb/dp_gamma_prior.asciipb \
    "" \
    resources/csv/in/multi_data.csv \
    resources/csv/in/multi_grid.csv \
    resources/csv/out/multi_dens.csv \
    resources/csv/out/multi_nclu.csv \
    resources/csv/out/multi_clus.csv
elif [ "$1" == 'lru' ]; then
  build/run \
    algo_marg_settings.asciipb \
    LinRegUni resources/asciipb/lin_reg_uni_fixed.asciipb \
    DP        resources/asciipb/dp_gamma_prior.asciipb \
    "" \
    resources/csv/in/lru_data.csv \
    resources/csv/in/lru_grid.csv \
    resources/csv/out/lru_dens.csv \
    resources/csv/out/lru_nclu.csv \
    resources/csv/out/lru_clus.csv \
    resources/csv/in/lru_hier_cov.csv \
    resources/csv/in/lru_hier_cov_grid.csv
elif [ "$1" == 'lsb' ]; then
  build/run \
    algo_cond_settings.asciipb \
    NNIG  resources/asciipb/nnig_ngg_prior.asciipb \
    LogSB resources/asciipb/lsb_normal_prior.asciipb \
    "" \
    resources/csv/in/logsb_data.csv \
    resources/csv/in/logsb_grid.csv \
    resources/csv/out/logsb_dens.csv \
    resources/csv/out/logsb_nclu.csv \
    resources/csv/out/logsb_clus.csv \
    "" \
    "" \
    resources/csv/in/logsb_cov_mix.csv \
    resources/csv/in/logsb_grid_cov_mix.csv
elif [ "$1" == 'tsb' ]; then
  build/run \
    algo_cond_settings.asciipb \
    NNIG    resources/asciipb/nnig_ngg_prior.asciipb \
    TruncSB resources/asciipb/truncsb_py_prior.asciipb \
    "" \
    resources/csv/in/truncsb_data.csv \
    resources/csv/in/truncsb_grid.csv \
    resources/csv/out/truncsb_dens.csv \
    resources/csv/out/truncsb_nclu.csv \
    resources/csv/out/truncsb_clus.csv \
    "" \
    "" \
    resources/csv/in/truncsb_cov_mix.csv \
    resources/csv/in/truncsb_grid_cov_mix.csv
else
  echo 'Syntax: bash/run_test.sh followed by uni, multi, lru, lsb, or tsb'
fi
