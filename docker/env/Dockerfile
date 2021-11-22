FROM ubuntu:latest

# Update apps on the base image
RUN apt-get -y update && apt-get install -y


RUN apt-get -y install g++ autoconf automake libtool curl make unzip 
RUN apt-get -y install ccache
RUN apt-get -y install python3-pip
RUN python3 -m pip install cmake
RUN DEBIAN_FRONTEND=noninteractive apt-get install -yq pkg-config
RUN apt-get -y install git

RUN apt-get -y install wget
RUN wget https://github.com/protocolbuffers/protobuf/releases/download/v3.14.0/protobuf-python-3.14.0.zip
RUN unzip protobuf-python-3.14.0.zip
RUN cd protobuf-3.14.0/ && ./configure --prefix=/usr && make install
RUN ldconfig

LABEL Name=bayesmix_env Version=0.0.1
