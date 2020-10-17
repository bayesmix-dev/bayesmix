#!/usr/bin/env bash

# Install necessary tools
sudo apt install autoconf automake libtool curl g++ unzip -y

# Get libraries
cd lib
git clone https://github.com/protocolbuffers/protobuf.git
git clone https://github.com/stan-dev/math.git

# Compile protobuf library
cd protobuf-3.12.3
./autogen.sh
./configure
make
make check
sudo make install
sudo ldconfig
cd ../..

# Install Python tools
python3 -m pip install matplotlib pybind11 scipy scikit-learn
python3 -m pip install --user --upgrade google-api-python-client
