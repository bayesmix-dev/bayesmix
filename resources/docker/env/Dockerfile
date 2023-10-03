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

LABEL Name=bayesmix-env Version=0.0.1
