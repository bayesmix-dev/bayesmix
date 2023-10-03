FROM mberaha/bayesmix-base:latest

# Store current version in /usr/bayesmix/update
COPY . ./update

# Generate patch file to update bayesmix
RUN cd update \
    && git remote add mainRepo https://github.com/bayesmix-dev/bayesmix.git \
    && git fetch mainRepo master \
    && git diff mainRepo/master > ../update.patch \
    && cd .. && rm -rf update

# Update only necessary files via patch
RUN if [ -s update.patch ] ; then git apply update.patch && rm update.patch ; fi

# Make test_bayesmix and run_mcmc after apply changes
RUN cd build \
    && cmake -DDISABLE_PLOTS=ON .. \
    && make test_bayesmix \
    && make run_mcmc

# # Export BAYESMIX_EXE variable
# ENV BAYESMIX_EXE=/usr/bayesmix/build/run_mcmc

# Install bayesmixpy
RUN cd python && python3 -m pip install -e .

# Install bayesmixr
# RUN cd R && Rscript --vanilla -e "devtools::install('bayesmixr/', quick = T, args = '--clean')"

LABEL Name=bayesmix-test Version=0.0.1
