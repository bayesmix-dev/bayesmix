<img src="resources/logo_full.svg" alt="drawing" width="250"/>

[![Documentation Status](https://readthedocs.org/projects/bayesmix/badge/?version=latest)](https://bayesmix.readthedocs.io/en/latest/?badge=latest)

`bayesmix` is a C++ library for running MCMC simulations in Bayesian mixture models.

Current state of the software:

- `bayesmix` performs inference for mixture models of the kind

```math
\begin{align*}
  y_1, \dots, y_n &\sim \int k\left(\cdot \mid \theta\right) P\left(\text{d}\theta\right) \\[3pt]
  P &\sim \Pi
\end{align*}
```

For descriptions of the models supported in our library, discussion of software design, and examples, please refer to the following paper: https://arxiv.org/abs/2205.08144

- Two lightweight interfaces are available for `Python` ([BayesMixPy](python/README.md)) and `R` (BayesMixR)
<!-- Add link after merge !!! -->

# Installation

For detailed instructions according to your operating system please refer to [INSTALL.md](INSTALL.md).

If all requirements are satisfied, to install and use `bayesmix`, please `cd` to the folder to which you wish to install it, and clone this repository with the following command-line instruction:

```shell
git clone https://github.com/bayesmix-dev/bayesmix.git
```

Then, by using `cd bayesmix`, you will enter the newly downloaded folder.

To build the executable for the main file `run_mcmc.cc`, please use the following list of commands:

```shell
mkdir build
cd build
cmake .. -DDISABLE_TESTS=ON
make run_mcmc
cd ..
```

# Tutorial

The `build/run_mcmc` executable can be used to perform all the necessary analysis, but it needs some command-line arguments to be passed.
To perform your first run of the library right out of the box, you can call the following script from the command line:

```shell
examples/tutorial/run.sh
```

This is an example script that runs said executable by passing appropriate arguments to it.
In order to use your custom datasets, algorithm settings, and prior specifications, you can create a copy of the above script and change the arguments as appropriate.
Please refer to the [documentation](#Documentation) for more information.

# Documentation

Documentation is available at https://bayesmix.readthedocs.io.

To build the documentation locally, make sure to have installed `Doxygen`, `sphinx`, and `docker`. Then

```shell
cd build
cmake .. -DENABLE_DOCS=ON
make document_bayesmix
```

will generete Sphinx documentation files in `build/docs/sphinx`. You can use a web browser to open the file `index.html`.

# Contributions are welcome!

Please check out [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to collaborate with us.
You can also head to our [issues page](https://github.com/bayesmix-dev/bayesmix/issues) to check for useful enhancements needed.
