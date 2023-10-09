FROM mberaha/bayesmix-base:latest

# Pull master branch to update bayesmix
RUN git pull

# Set working directory to /usr
WORKDIR /usr

# Store current version in /usr/bayesmix-update
COPY . bayesmix-update

# Generate and apply patch to updtae bayesmix
RUN diff -ruN -x 'build' -x '_deps' -x '.git' -x '*_pb2.py' bayesmix/ bayesmix-update/ | patch -d bayesmix -p1 \
    && rm -rf bayesmix-update

# Set working directory to /usr
WORKDIR /usr/bayesmix

# # Compile test_bayesmix and run_mcmc after apply changes
# RUN cd build \
#     && cmake -DDISABLE_PLOTS=ON .. \
#     && make test_bayesmix \
#     && make run_mcmc

# # Install bayesmixpy
# RUN cd python && python3 -m pip install -e .

# Install bayesmixr
# RUN cd R && Rscript --vanilla -e "devtools::install('bayesmixr/', quick = T, args = '--clean')"

LABEL Name=bayesmix-test Version=0.0.1
