#!/usr/bin/env bash

PROTO_DIR="./proto"

for filename in $PROTO_DIR/*.proto; do
  lib/protobuf-3.12.3/src/protoc --proto_path=$PROTO_DIR \
  --cpp_out=$PROTO_DIR/cpp/ $filename
done
#lib/protobuf-3.12.3/src/protoc --python_out=src/python chain_state.proto
