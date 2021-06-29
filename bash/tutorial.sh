#!/usr/bin/env bash

build/run \
  resources/tutorial/algo.asciipb \
  NNIG resources/tutorial/nnig_ngg.asciipb \
  DP   resources/tutorial/dp_gamma.asciipb \
  "" \
  resources/tutorial/data.csv \
  resources/tutorial/grid.csv \
  resources/tutorial/out/density.csv \
  resources/tutorial/out/numclust.csv \
  resources/tutorial/out/clustering.csv
