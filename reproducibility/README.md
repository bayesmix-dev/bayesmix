# Reproducibility Script

It is assumed that this folder `reproducibility` is located in the root folder. Imports won't work if other locations are chosen.
Moreover, it is required that the `bayesmixpy` interface is installed. See the "Installation" section of the README file in the `python/` folder for further information.

### Command Line Example

To reproduce the command line example from Section 5.2 in the paper, follow these instructions.

1. From within the root folder, create a build directory: `mkdir build`, and cd into it: `cd build`
2. run cmake: `cmake ..`
3. create the run and plot executables: `make run_mcmc; make plot_mcmc`
4. create the output directories: `mkdir reproducibility/log; mkdir reproducibility/csv; mkdir reproducibility/png;`
5. go back to the root folder: `cd ..`
6. run

```
./build/run_mcmc \
    --algo-params-file reproducibility/algo.asciipb \
    --hier-type NNIG --hier-args reproducibility/g0_params.asciipb \
    --mix-type DP --mix-args reproducibility/dp_param.asciipb \
    --coll-name reproducibility/log/chains.recordio \
    --data-file reproducibility/data.csv \
    --grid-file reproducibility/grid.csv \
    --dens-file reproducibility/csv/cmdline_density.csv \
    --n-cl-file reproducibility/csv/cmdline_numclust.csv \
    --clus-file reproducibility/csv/cmdline_clustering.csv \
    --best-clus-file reproducibility/csv/cmdline_best_clustering.csv
```

followed by

```
build/plot_mcmc \
    --grid-file resources/tutorial/grid.csv \
    --dens-file reproducibility/csv/cmdline_density.csv \
    --dens-plot reproducibility/png/cmdline_density.eps \
    --n-cl-file reproducibility/csv/cmdline_numclust.csv \
    --n-cl-trace-plot reproducibility/png/cmdline_traceplot.eps \
    --n-cl-bar-plot  reproducibility/png/cmdline_nclus_barplot.eps \
```

This will create the plots in the `reproducibility/png/` folder.

### Python Example

Please install the following python packages

```
arviz
matplotlib
numpy
pandas
```

and then run the reproducibility script as follows:

```
python3 -m reproducibility.replicate_jss_paper
```

All produced plots will be saved to the `reproducibility/png/` folder.
