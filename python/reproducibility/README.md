# Reproducibility Script

It is assumed that this folder `reproducibility` is located inside `python/`. Imports won't work if other locations are chosen.
Moreover, it is required that the `bayesmixpy` interface is installed. See the "Installation" section of the README file in the `python/` folder for further information.

To reproduce all plots from the paper, please `cd` into the `python/` folder and run the reproducibility script as follows:

```
python3 -m reproducibility.replicate_jss_paper
```

All produced plots will be saved to the `reproducibility/png/` folder.
