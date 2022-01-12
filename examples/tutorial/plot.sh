#!/usr/bin/env bash

build/plots/plot_mcmc \
  --grid-file resources/tutorial/grid.csv \
  --dens-file resources/tutorial/out/density_file.csv \
  --n-cl-file resources/tutorial/out/numclust.csv \
  --dens-plot resources/tutorial/out/density.png \
  --n-cl-trace-plot resources/tutorial/out/traceplot.png \
  --n-cl-hist-plot  resources/tutorial/out/histogram.png
