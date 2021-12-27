FROM mberaha/bayesmix-base:latest

COPY cmake /usr/bayesmix/cmake
COPY proto /usr/bayesmix/proto
COPY resources /usr/bayesmix/resources
COPY src /usr/bayesmix/src
COPY test /usr/bayesmix/test
COPY CMakeLists.txt /usr/bayesmix/CMakeLists.txt
COPY *.cc /usr/bayesmix/

WORKDIR /usr/bayesmix

RUN ls lib/math/lib/tbb
RUN ls /usr/bayesmix/lib/math/lib/tbb/

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/bayesmix/lib/math/lib/tbb/

RUN cd build_ && cmake -DDISABLE_DOCS=ON -DDISABLE_BENCHMARKS=ON -DBUILD_RUN=OFF .. && make test_bayesmix

CMD ["./build_/test/test_bayesmix"]

LABEL Name=test_bayesmix Version=0.0.1
