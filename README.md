bayesmix is a C++ library for running MCMC simulation in Bayesian mixture models.

Current state of the software:
- bayesmix performs inference for mixture models of the kind

<a href="https://www.codecogs.com/eqnedit.php?latex=y_1,&space;\ldots,&space;y_n&space;\sim&space;\int&space;k(\cdot&space;\mid&space;\theta)&space;P(d\theta)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y_1,&space;\ldots,&space;y_n&space;\sim&space;\int&space;k(\cdot&space;\mid&space;\theta)&space;P(d\theta)" title="y_1, \ldots, y_n \sim \int k(\cdot \mid \theta) \Pi(d\theta)" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=P&space;\sim&space;\Pi" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P&space;\sim&space;\Pi" title="\Pi \sim P" /></a>

Where P is either the Dirichlet process or the Pitman--Yor process.

- We currently support univariate and multivariate location-scale mixture of Gaussian densities

- Inference is carried out using either Algorithm 2 or Algorithm 8 in [Neal (2000)](http://www.stat.columbia.edu/npbayes/papers/neal_sampling.pdf).

- Serialization of the MCMC chains is possible using [Google's protocol buffers](https://developers.google.com/protocol-buffers)


## Installation:

We heavily depend on Google's [Protocol Buffers](https://github.com/protocolbuffers/protobuf), so make sure to install it beforehand!

On Linux machine the following will install the library
```shell
sudo apt-get install autoconf automake libtool curl make g++ unzip
wget https://github.com/protocolbuffers/protobuf/releases/download/v3.14.0/protobuf-python-3.14.0.zip
unizp protobuf-python-3.14.0.zip
cd protobuf-3.14.0/
./configure --prefix=/usr
make check
sudo make install
sudo ldconfig # refresh shared library cache.
```
On Mac and Windows machines, follow the official install guide ([link](https://github.com/protocolbuffers/protobuf/blob/master/src/README.md))

Finally, to work with `bayesmix` just clone the repository with
```shell
git clone --recurse-submodule git@github.com:bayesmix-dev/bayesmix.git
```

To run the executable:
```shell
mkdir build
cd build
cmake ..
make run
cd ..
./build/run
```

To run unit tests:
```shell
cd build
cmake ..
make test_bayesmix
./test/test_bayesmix
```

## Cluster estimate

This library provides a cluster estimates computation, given a mcmc chains. 
It is based on expected posterior loss minimisation given a loss function and using a greedy algorithm.
Sources files are in the folder `src/clustering`.

To run the code :
```shell
cd build
cmake ..
make run_pe
./run_pe filename_in filename_out loss Kup
```

where :

- filename_in is the entry filename that contains mcmc chain (a file in which values are separated with spaces)
- filename_out is the out filename in which cluster estimate will be writen
- loss is the specification of the loss function : 0 for binder loss, 1 for variation of information, 2 for normalized variation of information
- Kup is the max number of clusters (usually Kup=N is a good entry if dataset has a length of N)

Credible balls computation is also available. This aims to quantify the uncertainty of a cluster estimate. 
To run the credible balls code : 

```shell
cd build
cmake ..
make run_cb
./run_cb filename_mcmc filename_pe filename_out loss rate
```

where :
- filename_mcmc is the filename in which there is the mcmc chain.
- filename_pe is the filename in which there is the cluster estimate.
- filename_out  is the filename in which result will be writen
- loss is the specification of the loss function : 0 for binder loss, 1 for variation of information, 2 for normalized variation of information
- rate : has to be > 0. The smaller it is, the longer will run the program.


The directory `src/clustering/R scripts` contains some scripts to generate mcmc chains for univariate and multivariate datasets.


## Contributions are welcome!
Please check out [CONTRIBUTORS.md](CONTRIBUTORS.md) for details on how to collaborate with us.
