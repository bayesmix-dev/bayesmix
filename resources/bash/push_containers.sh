#! /bin/sh
docker build -f docker/env/Dockerfile --platform linux/x86_64 -t mberaha/bayesmix-env .
docker push mberaha/bayesmix-env
docker build -f docker/base/Dockerfile --platform linux/x86_64 -t mberaha/bayesmix-base .
docker push mberaha/bayesmix-base
