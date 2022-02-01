#!/usr/bin/env bash

build/plots/plot_mcmc \
  --grid-file resources/datasets/faithful_grid.csv \
  --dens-file resources/2d/density_2d.csv \
  --dens-plot resources/2d/density.png \
  --n-cl-file resources/2d/numclust_2d.csv \
  --n-cl-trace-plot resources/2d/traceplot.png \
  --n-cl-bar-plot  resources/2d/nclus_barplot.png
