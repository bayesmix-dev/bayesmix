FROM mberaha/bayesmix-env:latest

# Clone bayesmix-dev/bayesmix repository in /usr/bayesmix
RUN git clone https://github.com/bayesmix-dev/bayesmix.git /usr/bayesmix

# Set working directory to /usr/bayesmix
WORKDIR /usr/bayesmix

# Make test_bayesmix and run_mcmc
RUN mkdir build && cd build \
    && cmake -DDISABLE_PLOTS=ON .. \
    && make test_bayesmix \
    && make run_mcmc

LABEL Name=bayesmix-base Version=0.0.1
