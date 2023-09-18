<img src="resources/logo_full.svg" alt="drawing" width="250"/>

[![Documentation Status](https://readthedocs.org/projects/bayesmix/badge/?version=latest)](https://bayesmix.readthedocs.io/en/latest/?badge=latest)

`bayesmix` is a C++ library for running MCMC simulations in Bayesian mixture models.

Current state of the software:

- `bayesmix` performs inference for mixture models of the kind
  ``` math
  \begin{align*}
   y_1, \dots, y_n &\sim \int k\left(\cdot \mid \theta\right) P\left(\text{d}\theta\right) \\[3pt]
   P &\sim \Pi
  \end{align*}
  ```
<!---
<img src="https://latex.codecogs.com/svg.image?\begin{align*}y_1,\dots,y_n&space;&\sim&space;\int&space;k(\cdot&space;\mid&space;\theta)P(\mathrm{d}\theta)\\P&space;&\sim&space;\Pi\end{align*}&space;" title="\begin{align*}y_1,\dots,y_n &\sim \int k(\cdot \mid \theta)P(\mathrm{d}\theta)\\P &\sim \Pi\end{align*} " />
-->

<!---
<a href="https://www.codecogs.com/eqnedit.php?latex=y_1,&space;\ldots,&space;y_n&space;\sim&space;\int&space;k(\cdot&space;\mid&space;\theta)&space;P(d\theta)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y_1,&space;\ldots,&space;y_n&space;\sim&space;\int&space;k(\cdot&space;\mid&space;\theta)&space;P(d\theta)" title="y_1, \ldots, y_n \sim \int k(\cdot \mid \theta) \Pi(d\theta)" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=P&space;\sim&space;\Pi" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P&space;\sim&space;\Pi" title="\Pi \sim P" /></a>
-->

For descriptions of the models supported in our library, discussion of software design, and examples, please refer to the following paper: https://arxiv.org/abs/2205.08144

- Two lightweight interfaces are available for `Python` ([BayesMixPy](https://github.com/bayesmix-dev/bayesmix/tree/master/python#readme)) and `R` (BayesMixR)
<!-- Add link after merge !!! -->

# Installation
Compiling and building `bayesmix` requires a modern C++ compiler, `git`, the GNU `make` utility, `cmake` (version >= 3.20) and `pkg-config`. Make these packages available varies between operating systems.  

**Warning**: Check the version of `cmake` available in your package manager default repository. You may need to install a more recent version in another way. If this is your case, please go to the [official CMake GitHub repository](https://github.com/Kitware/CMake) and follow the installation instructions.

## Requirements - Linux
On Linux, the required packages can be installed via the default package manager. For instance, in Ubuntu it is sufficient to run

```shell
sudo apt install git
sudo apt install g++
sudo apt install make
sudo apt install cmake
sudo apt install pkg-config
```

## Requirements - MacOS
On MacOS, the required packages can be installed via [Homebrew](https://brew.sh/). Once installed, it is sufficient to run

```shell
brew install git
brew install g++
brew install make
brew install cmake
brew install pkg-config
```

## Requirements - Windows
First of all, install `git` via the [Git for Windows](https://gitforwindows.org/) project. Download the [installer](https://github.com/git-for-windows/git/releases/latest) and complete the prompts leaving default choices to install. The Git BASH that comes with this program is the BASH we suggest to compile and run `bayesmix`.  

On Windows, we also need the installation of a proper C++ toolchain to install the other required packages. `bayesmix` is known complatible with RTools40, RTools42 and RTools43 toolchains. These require slightly different steps to configure, so please follow the appropriate steps below. All toolchains will require updating your `PATH` variable, See [these instructions](https://helpdeskgeek.com/windows-10/add-windows-path-environment-variable/) for details on changing the `PATH` if you are unfamiliar. The following instructions will assume that the default installation directory was used, so be sure to update the paths accordingly if you have chosen a different directory.

### Configure RTools40
Download the [installer](https://github.com/r-windows/rtools-installer/releases/download/2022-02-06/rtools40-x86_64.exe) and complete the prompts to install.
Next, add the following lines to your `PATH`:

```shell
C:\rtools40\usr\bin
C:\rtools40\mingw64\bin
```
Then, the other dependencies can be installed by typing the following commands into one of the shells installed with RTools (e.g. lauch the `C:\rtools40\msys.exe` file)

```shell
pacman -Sy mingw-w64-x86_64-gcc
pacman -Sy mingw-w64-x86_64-make
pacman -Sy mingw-w64-x86_64-cmake
pacman -Sy mingw-w64-x86_64-pkg-config
```
### Configure RTools42 / RTools43
These two versions of RTools toolchain are quite similar. Of course, RTools43 offers newer version of the packages, but in both cases the installation and configuration is identical.
Download either the [RTools42 installer](https://cran.r-project.org/bin/windows/Rtools/rtools42/files/rtools42-5355-5357.exe) or the [RTools43 installer](https://cran.r-project.org/bin/windows/Rtools/rtools43/files/rtools43-5550-5548.exe) and complete the prompts to install.
Next, add the following lines to your `PATH`:

```shell
# RTools42
C:\rtools42\usr\bin
C:\rtools42\...\bin
C:\rtools42\ucrt64\bin

# RTools43
C:\rtools43\usr\bin
C:\rtools43\...\bin
C:\rtools43\ucrt64\bin
```

Then, the other dependencies can be installed by typing the following commands into one of the shells installed with RTools (e.g. lauch the `C:\rtools42\msys.exe` or `C:\rtools43\msys.exe` file)

```shell
pacman -Sy mingw-w64-ucrt-x86_64-gcc
pacman -Sy mingw-w64-ucrt-x86_64-make
pacman -Sy mingw-w64-ucrt-x86_64-cmake
pacman -Sy mingw-w64-ucrt-x86_64-pkgconf
```

## Build `bayesmix`
**Note for Windows**: Use the Git BASH shell available with Git for Windows to execute these commands. If `PATH` has been configured correctly, all requirements will be satisfied.  

To install and use `bayesmix`, please `cd` to the folder to which you wish to install it, and clone this repository with the following command-line instruction:

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

Finally, to compile unit tests, please use the following commands:

```shell
cd build
cmake ..
make test_bayesmix
cd ..
```

The corresponding executable is located at `build/test/test_bayesmix`.

<!---
## For end users

**Prerequisites**: to build `bayesmix` you will need `gihttps://cran.r-project.org/bin/windows/Rtools/rtools43/files/rtools43-5550-5548.exet`, `pkg-config` and a recent version of `cmake`.
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
cmake .. -DDISABLE_TESTS=ON
make run_mcmc
cd ..
```

--->

### Tutorial

The `build/run_mcmc` executable can be used to perform all the necessary analysis, but it needs some command-line arguments to be passed.
To perform your first run of the library right out of the box, you can call the following script from the command line:

```shell
examples/tutorial/run.sh
```

This is an example script that runs said executable by passing appropriate arguments to it.
In order to use your custom datasets, algorithm settings, and prior specifications, you can create a copy of the above script and change the arguments as appropriate.
Please refer to the [documentation](#Documentation) for more information.

<!---
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
--->

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
