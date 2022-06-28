FROM mberaha/bayesmix-base

COPY cmake /usr/bayesmix/cmake
COPY resources /usr/bayesmix/resources
COPY src /usr/bayesmix/src
COPY test /usr/bayesmix/test
COPY python /usr/bayesmix/python
COPY CMakeLists.txt /usr/bayesmix/CMakeLists.txt
COPY lib/argparse /usr/bayesmix/lib/argparse
COPY executables/ /usr/bayesmix/executables

WORKDIR /usr/bayesmix

RUN python3 -m pip install pytest

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/bayesmix/lib/math/lib/tbb/

RUN cd build_ && cmake -DDISABLE_DOCS=ON -DDISABLE_BENCHMARKS=ON -DDISABLE_PLOTS=ON .. && make test_bayesmix
RUN cd build_ && make run_mcmc

ENV BAYESMIX_EXE=/usr/bayesmix/build_/run_mcmc
RUN cd python && python3 -m pip install -e .

LABEL Name=test_bayesmix Version=0.0.1
