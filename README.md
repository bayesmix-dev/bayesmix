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

After cloning this git repository, run 
```shell
./bash/install_libs.sh
```

This will install the [Stan math library](https://github.com/stan-dev/math) and Protocol buffers in `lib`

To run the executable:
```shell
mkdir build
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

## Future steps (contributors are welcome!)

A Python package is already under development

- Extension to normalized random measures
- Using HMC / MALA MCMC algorithm to sample from the cluster-specific full conditionals when it's not conjugate to the base measure
- R package



