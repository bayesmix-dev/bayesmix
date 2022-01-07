FROM mberaha/bayesmix-base

COPY cmake /usr/bayesmix/cmake
COPY proto /usr/bayesmix/proto
COPY resources /usr/bayesmix/resources
COPY src /usr/bayesmix/src
COPY test /usr/bayesmix/test
COPY CMakeLists.txt /usr/bayesmix/CMakeLists.txt
COPY lib/argparse /usr/bayesmix/lib/argparse
COPY *.cc /usr/bayesmix/

WORKDIR /usr/bayesmix

RUN ls lib/

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/bayesmix/lib/math/lib/tbb/

RUN cd build_ && cmake -DDISABLE_DOCS=ON -DDISABLE_BENCHMARKS=ON .. && make test_bayesmix
RUN cd build_ && make run_mcmc

CMD ["./build_/test/test_bayesmix; cd python; BAYESMIX_EXE=../build_/run_mcmc pytest"]

LABEL Name=test_bayesmix Version=0.0.1
