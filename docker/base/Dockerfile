FROM mberaha/bayesmix-env:latest

COPY . /usr/bayesmix
WORKDIR /usr/bayesmix

RUN python3 build_tbb.py --rebuild=True

RUN rm -rf build_

RUN mkdir build_ && cd build_ && cmake -DDISABLE_DOCS=ON -DDISABLE_BENCHMARKS=ON .. && make test_bayesmix
CMD ["./build_/test/test_bayesmix"]

LABEL Name=bayesmix_base Version=0.0.1
