#!/usr/bin/env bash

build/run_mcmc \
  --algo_params_file resources/tutorial/algo.asciipb \
  --hier_type NNIG --hier_args resources/tutorial/nnig_ngg.asciipb \
  --mix_type DP --mix_args resources/tutorial/dp_gamma.asciipb \
  --collname resources/tutorial/out/chains.recordio \
  --datafile resources/tutorial/data.csv \
  --gridfile resources/tutorial/grid.csv \
  --densfile resources/tutorial/out/density_file.csv \
  --nclufile resources/tutorial/out/numclust.csv \
  --clusfile resources/tutorial/out/clustering.csv
