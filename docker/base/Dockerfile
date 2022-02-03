FROM mberaha/bayesmix-env:latest

COPY . /usr/bayesmix
WORKDIR /usr/bayesmix

RUN python3 -m pip install pytest

RUN mkdir build_ && cd build_ && cmake -DDISABLE_DOCS=ON -DDISABLE_BENCHMARKS=ON -DDISABLE_PLOTS=ON .. && make test_bayesmix
RUN cd build_ && make run_mcmc

CMD ["./build_/test/test_bayesmix"]

LABEL Name=bayesmix_base Version=0.0.1
