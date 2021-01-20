#!/usr/bin/env bash

PROTO_DIR="./proto"

for filename in $PROTO_DIR/*.proto; do
  protoc --proto_path=$PROTO_DIR \
  --cpp_out=$PROTO_DIR/cpp/ $filename
  protoc --proto_path=$PROTO_DIR \
  --python_out=$PROTO_DIR/py/ $filename
done

2to3 --output-dir=$PROTO_DIR/py/ -W -n $PROTO_DIR/py/
