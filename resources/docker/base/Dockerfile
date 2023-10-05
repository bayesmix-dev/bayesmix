FROM mberaha/bayesmix-env:latest

# Pull master branch to update bayesmix
RUN git pull

# Compile test_bayesmix and run_mcmc
RUN cd build \
    && cmake -DDISABLE_PLOTS=ON .. \
    && make test_bayesmix \
    && make run_mcmc

LABEL Name=bayesmix-base Version=0.0.1
