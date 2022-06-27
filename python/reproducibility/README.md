# Reproducibility Scripts

It is assumed that this folder `reproducibility` is located inside `python/`. Imports won't work if other locations are chosen.
Moreover, it is required that the `bayesmixpy` interface is installed, by opening the terminal in the `python/` folder then running:

```
python3 -m pip install -e .
```

If the above command does not work, try again replacing `-e` with `-U`.

- To reproduce Figures 2 and 3 of the paper, run the `python_example.ipynb` notebook
- To reproduce Figure 4 and Tables 3, 4, and 5, first run `generate_high_dim_data.py`, then `run_bnpmix.R` and finally `compare.ipynb`
- To reproduce Figure 1, open the terminal at the root folder of `bayesmix`, then execute the following commands:

```
mkdir build
cd build
cmake ..
make run_mcmc -j4
make plot_mcmc
cd ..
```

(or any number of cores after `-j`), then run:

```
./examples/tutorial/run.sh
./examples/tutorial/plot.sh
```

Figures from the tutorial files will be saved in the folder `resources/tutorial/out`.
