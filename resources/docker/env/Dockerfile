FROM rocker/r-ubuntu:latest

# Update repo and install required packages
RUN apt-get -y update \
    && apt-get -y upgrade \
    && apt-get -y install ccache cmake g++ git make pkg-config

# Install required python packages
RUN apt-get -y install python3-pip \
    && python3 -m pip install pytest

# Install required R packages (also installing protobuf and protoc v. 3.12)
RUN apt-get -y install r-cran-devtools r-cran-testthat r-cran-rprotobuf

# Clone bayesmix-dev/bayesmix repository in /usr/bayesmix
RUN git clone https://github.com/bayesmix-dev/bayesmix.git /usr/bayesmix

# Set working directory to /usr/bayesmix
WORKDIR /usr/bayesmix

# Compile test_bayesmix and run_mcmc
RUN mkdir build && cd build \
    && cmake -DDISABLE_PLOTS=ON .. \
    && make test_bayesmix \
    && make run_mcmc

LABEL Name=bayesmix-env Version=0.0.1
