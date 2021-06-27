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

### Library
To work with `bayesmix`, first clone the repository with:
```shell
git clone --recurse-submodule git@github.com:bayesmix-dev/bayesmix.git
```
Then, `cd` into the `bayesmix` root folder and compile the library executable as follows:
```shell
mkdir build
cd build
cmake ..
make run
```
If the `cmake` line returns errors about not finding the GTest/GoogleTest library, please run the following line in its place:
```shell
cmake .. -DDISABLE_TESTS="ON" -DDISABLE_BENCHMARKS="ON"
```

To compile and run unit tests, use:
```shell
cd build
make test_bayesmix
./test/test_bayesmix
```

## Usage
You can refer to our documentation for more in-depth information about this library.
You can find it at https://bayesmix.readthedocs.io, or you can compile in on your machine by running:
```shell
cd build
make Sphinx
```
Then, navigate to the `build/docs/sphinx/index.html` file and open it with your favorite browser/HTML reader.

### Examples
TODO coming soon!

## Contributions are welcome!
Please check out [CONTRIBUTORS.md](CONTRIBUTORS.md) for details on how to collaborate with us.
