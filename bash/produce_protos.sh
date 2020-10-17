#!/usr/bin/env bash

lib/protobuf-3.12.3/src/protoc --cpp_out=src/collectors chain_state.proto
#lib/protobuf-3.12.3/src/protoc --python_out=src/python chain_state.proto
