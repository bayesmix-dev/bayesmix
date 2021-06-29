<img src="../resources/logo_full.svg" alt="drawing" width="250"/>

[![Documentation Status](https://readthedocs.org/projects/bayesmix/badge/?version=latest)](https://bayesmix.readthedocs.io/en/latest/?badge=latest)

`bayesmix` is a C++ library for running MCMC simulation in Bayesian mixture models.

Current state of the software:

- `bayesmix` performs inference for mixture models of the kind

<a href="https://www.codecogs.com/eqnedit.php?latex=y_1,&space;\ldots,&space;y_n&space;\sim&space;\int&space;k(\cdot&space;\mid&space;\theta)&space;P(d\theta)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y_1,&space;\ldots,&space;y_n&space;\sim&space;\int&space;k(\cdot&space;\mid&space;\theta)&space;P(d\theta)" title="y_1, \ldots, y_n \sim \int k(\cdot \mid \theta) \Pi(d\theta)" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=P&space;\sim&space;\Pi" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P&space;\sim&space;\Pi" title="\Pi \sim P" /></a>

where P is either the Dirichlet process or the Pitman-Yor process.

- We currently support univariate and multivariate location-scale mixture of Gaussian densities

- Inference is carried out using algorithms such as Neal's Algorithm 2 from [Neal (2000)](http://www.stat.columbia.edu/npbayes/papers/neal_sampling.pdf)

- Serialization of the MCMC chains is possible using [Google's protocol buffers](https://developers.google.com/protocol-buffers).



## Installation
### Dependencies
We heavily depend on Google's [Protocol Buffers](https://github.com/protocolbuffers/protobuf), so make sure to install it beforehand! In particular, on a Linux machine the following will install the `protobuf` library:
```shell
sudo apt-get install autoconf automake libtool curl make g++ unzip cmake
wget https://github.com/protocolbuffers/protobuf/releases/download/v3.14.0/protobuf-python-3.14.0.zip
unzip protobuf-python-3.14.0.zip
cd protobuf-3.14.0/
./configure --prefix=/usr
make check
sudo make install
sudo ldconfig
```
On Mac and Windows machines, please follow the [official install guide](https://github.com/protocolbuffers/protobuf/blob/master/src/README.md) instead.

We also depend on popular ``Eigen`` and ``Stan`` libraries for vector-matrix operations and probability functions, respectively, but these libraries come included with `bayesmix` so you do not need to install them separately.

### Library
To work with `bayesmix`, first clone the repository with:
```shell
git clone --recurse-submodule git@github.com:bayesmix-dev/bayesmix.git
```
Then, `cd` into the `bayesmix` root folder and compile the library executable as follows:
```shell
mkdir build
cd build
cmake .. -DDISABLE_DOCS="ON"
make run
```
If the `cmake` line returns errors about not finding the GTest/GoogleTest library, please add `-DDISABLE_TESTS="ON" -DDISABLE_BENCHMARKS="ON"` at the end of the command and try again.
```

To compile and run unit tests, use:
```shell
cd build
make test_bayesmix
./test/test_bayesmix
```



## Usage
You can refer to our documentation for more in-depth information about this library.
You can find it at https://bayesmix.readthedocs.io, or you can compile in on your machine by first installing required dependences:
```shell
sudo apt install doxygen python3-pip breathe
python3 -m pip install sphinx_rtd_theme
```
then `cd`ing into the root folder and running:
```
cd build
make Sphinx
```
Then, navigate to the `build/docs/sphinx/index.html` file and open it with your favorite browser/HTML reader.

### Examples
`run.cc` is an example of C++ main file that performs MCMC simulation and density estimation via `bayesmix`.
It needs a few command line parameters, as detailed below:
```shell
build/run \
  algorithm_parameters_file \
  hierarchy_type hierarchy_args \
  mixing_type mixing_args \
  collector_name \
  data_file \
  grid_file \
  density_file \
  num_clusters_file \
  clustering_file \
  [hierarchy_covariates_file] \
  [hierarchy_grid_covariates_file] \
  [mixing_covariates_file] \
  [mixing_grid_covariates_file]
```
First batch of parameters up until `grid_file` is mainly composed of names of input files or objects IDs. Then there are output files, and finally, in square brackets, optional input files, in the case that the chosen hierarchy needs covariates.
A working example follows, taken from `bash/tutorial.sh`, which uses input files already present in the library:
```shell
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

```
Due to the large number of parameters, it is recommended that such a command is written to a `.sh` script which is then executed, just like with the given example.



## Contributions are welcome!
Please check out [CONTRIBUTORS.md](CONTRIBUTORS.md) for details on how to collaborate with us.
