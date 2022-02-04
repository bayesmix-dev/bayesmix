#!/usr/bin/env bash

build/run_mcmc \
  --algo-params-file resources/tutorial/algo.asciipb \
  --hier-type NNW --hier-args resources/tutorial/nnw_ngiw.asciipb \
  --mix-type DP --mix-args resources/tutorial/dp_gamma.asciipb \
  --data-file resources/datasets/faithful.csv \
  --grid-file resources/datasets/faithful_grid.csv \
  --coll-name resources/2d/chains_2d.recordio \
  --dens-file resources/2d/density_2d.csv \
  --n-cl-file resources/2d/numclust_2d.csv \
  --clus-file resources/2d/clustering_2d.csv
