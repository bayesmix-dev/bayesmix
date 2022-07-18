<img src="resources/logo_full.svg" alt="drawing" width="250"/>

[![Documentation Status](https://readthedocs.org/projects/bayesmix/badge/?version=latest)](https://bayesmix.readthedocs.io/en/latest/?badge=latest)

`bayesmix` is a C++ library for running MCMC simulations in Bayesian mixture models.

Current state of the software:

- `bayesmix` performs inference for mixture models of the kind

<img src="https://latex.codecogs.com/svg.image?\begin{align*}y_1,\dots,y_n&space;&\sim&space;\int&space;k(\cdot&space;\mid&space;\theta)P(\mathrm{d}\theta)\\P&space;&\sim&space;\Pi\end{align*}&space;" title="\begin{align*}y_1,\dots,y_n &\sim \int k(\cdot \mid \theta)P(\mathrm{d}\theta)\\P &\sim \Pi\end{align*} " />

<!---
<a href="https://www.codecogs.com/eqnedit.php?latex=y_1,&space;\ldots,&space;y_n&space;\sim&space;\int&space;k(\cdot&space;\mid&space;\theta)&space;P(d\theta)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y_1,&space;\ldots,&space;y_n&space;\sim&space;\int&space;k(\cdot&space;\mid&space;\theta)&space;P(d\theta)" title="y_1, \ldots, y_n \sim \int k(\cdot \mid \theta) \Pi(d\theta)" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=P&space;\sim&space;\Pi" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P&space;\sim&space;\Pi" title="\Pi \sim P" /></a>
-->

For descriptions of the models supported in our library, discussion of software design, and examples, please refer to the following paper: https://arxiv.org/abs/2205.08144

# Installation

## For end users

**Prerequisites**: to build `bayesmix` you will need `git`, `pkg-config` and a recent version of `cmake`.
On Linux machines, it is sufficient to run

```shell
 sudo apt-get -y update && apt-get install -y
 sudo apt-get -y install git
 sudo apt-get -y install python3-pip
 sudo python3 -m pip install cmake
 sudo apt-get install -yq pkg-config
```

On macOS, after install HomeBrew, replace `sudo apt-get -y` with `brew`.

To install and use `bayesmix`, please `cd` to the folder to which you wish to install it, and clone this repository with the following command-line instruction:

```shell
git clone https://github.com/bayesmix-dev/bayesmix.git
```

Then, by using `cd bayesmix`, you will enter the newly downloaded folder.

To build the executable for the main file `run_mcmc.cc`, please use the following list of commands:

```shell
mkdir build
cd build
cmake .. -DDISABLE_TESTS=ON -DDISABLE_BENCHMARKS=ON
make run_mcmc
cd ..
```

### Tutorial

The `build/run_mcmc` executable can be used to perform all the necessary analysis, but it needs some command-line arguments to be passed.
To perform your first run of the library right out of the box, you can call the following script from the command line:

```shell
examples/tutorial/run.sh
```

This is an example script that runs said executable by passing appropriate arguments to it.
In order to use your custom datasets, algorithm settings, and prior specifications, you can create a copy of the above script and change the arguments as appropriate.
Please refer to the [documentation](#Documentation) for more information.

## For developers

We heavily depend on the `protobuf` library to move and store structured data.
The `CMakeLists.txt` file is set up to install such library if it does not find it in the computer.
However any call to `make clean` will uninstall it, causing a huge waste of time... so make sure to install it manually beforehand!
If you're using a Linux machine, you can do so as follows:

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

On Mac and Windows machines, please follow the steps from the [official `protobuf` installation guide](https://github.com/protocolbuffers/protobuf/blob/master/src/README.md).

Another very useful tool is [`ccache`](https://ccache.dev) that can significantly speed up the compilation process.

Finally, to compile unit tests, please use the following commands:

```shell
cd build
cmake ..
make test_bayesmix
cd ..
```

The corresponding executable is located at `build/test/test_bayesmix`.

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
