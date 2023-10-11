# Installation

Compiling and building `bayesmix` requires a modern C++ compiler, `git`, the GNU `make` utility, `cmake` (version >= 3.16) and `pkg-config`. Make these packages available varies between operating systems.

**Warning**: Check the version of `cmake` available in your package manager default repository. This requirement is already satisfied in Ubuntu 20.04 LTS, so in fairly updated distros this requirement will already be satisfied. If your system is too old and the binary `cmake` package does not satisfy the minimum requiremens, please go to the [official CMake GitHub repository](https://github.com/Kitware/CMake) and follow the installation instructions.

## Requirements - Linux

On Linux systems, the required packages can be installed via the system's default package manager. For instance, in Ubuntu or other Debian based distros it is sufficient to run

```shell
sudo apt install git g++ make cmake pkg-config
```

For other Linux distributions, you need to adapt the above instruction to the syntax of your package manager. [This table](https://wiki.archlinux.org/title/Pacman/Rosetta) shows the correspondence of package management commands among some of the most popular Linux distributions.

## Requirements - MacOS

On MacOS, the required packages can be installed via [Homebrew](https://brew.sh/). Once installed, it is sufficient to run

```shell
brew install git g++ make cmake pkg-config
```

## Requirements - Windows

First of all, install `git` via the [Git for Windows](https://gitforwindows.org/) project. Download the [installer](https://github.com/git-for-windows/git/releases/latest) and complete the prompts leaving default choices to install.

On Windows, we also need the installation of a proper C++ toolchain and install the other required packages. `bayesmix` can be successfully compiled and installed with RTools40, RTools42 and RTools43 toolchains. This choice simplified the development on a lightweight `R` interface working on all platforms.

[Rtools](https://cran.r-project.org/bin/windows/Rtools/) is a toolchain bundle used for building `R` packages from source (those that need compilation of C/C++ or Fortran code) and for building `R` itself. Rtools usually consists of Msys2 build tools, GCC/MinGW-w64 compiler toolchain and libraries. These require slightly different steps to configure, so please follow the appropriate steps below.

All toolchains will require updating your `PATH` variable, See [these instructions](https://helpdeskgeek.com/windows-10/add-windows-path-environment-variable/) for details on changing the `PATH` if you are unfamiliar. The following instructions will assume that the default installation directory was used, so be sure to update the paths accordingly if you have chosen a different directory.

### Configure RTools40

Download the [installer](https://github.com/r-windows/rtools-installer/releases/download/2022-02-06/rtools40-x86_64.exe) and complete the prompts to install.
Next, add the following lines to your `PATH`:

```shell
C:\rtools40\usr\bin
C:\rtools40\mingw64\bin
```

The C++ compiler is now available on your system. All the other dependencies can be installed by typing the following commands into Windows Command Prompt or Windows PowerShell:

<!-- one of the shells installed with RTools (e.g. launch the `C:\rtools40\msys.exe` file) -->

```shell
pacman -Sy mingw-w64-x86_64-make
pacman -Sy mingw-w64-x86_64-cmake
pacman -Sy mingw-w64-x86_64-pkg-config
```

### Configure RTools42 / RTools43

These two versions of RTools toolchain are quite similar. Of course, RTools43 offers newer version of the packages, but in both cases the installation and configuration is identical.
Download either the [RTools42 installer](https://cran.r-project.org/bin/windows/Rtools/rtools42/files/rtools42-5355-5357.exe) or the [RTools43 installer](https://cran.r-project.org/bin/windows/Rtools/rtools43/files/rtools43-5550-5548.exe) and complete the prompts to install.
Next, add the following lines to your `PATH`:

#### RTools42

```shell
C:\rtools42\x86_64-w64-mingw32.static.posix\bin
C:\rtools42\ucrt64\bin
C:\rtools42\usr\bin
```

#### RTools43
```shell
C:\rtools43\x86_64-w64-mingw32.static.posix\bin
C:\rtools43\ucrt64\bin
C:\rtools43\usr\bin
```

The C++ compiler is now available on your system. All the other dependencies can be installed by typing the following commands into into Windows Command Prompt or Windows PowerShell:
<!--one of the shells installed with RTools (e.g. lauch the `C:\rtools42\msys.exe` or `C:\rtools43\msys.exe` file) -->

```shell
pacman -Sy mingw-w64-ucrt-x86_64-make
pacman -Sy mingw-w64-ucrt-x86_64-cmake
pacman -Sy mingw-w64-ucrt-x86_64-pkgconf
```

### Important remarks :

- Use the native Windows Command Prompt (or Windows PowerShell) to execute these commands. If `PATH` environment variable has been configured correctly, all requirements will be satisfied.
- In order for `bayesmix` to be properly linked to Intel's TBB library, the absolute path to `tbb` must be added to the User `PATH` variable. This is done automatically during build but to make this change effective user need to close and open a new <!-- Git BASH --> shell.

## Build `bayesmix`

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

**Remark**: Configuring and compiling `bayesmix` for the first time will require several minutes, since `cmake` needs to provide and compile all the libraries on which `bayesmix` depends.

Finally, to compile unit tests, please use the following commands:

```shell
cd build
cmake ..
make test_bayesmix
cd ..
```

The corresponding executable is located at `build/test/test_bayesmix`.
