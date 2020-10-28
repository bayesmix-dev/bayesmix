#!/bin/bash

PROTO_DIR="./proto"

for filename in $PROTO_DIR/*.proto; do
  protoc --proto_path=$PROTO_DIR --cpp_out=$PROTO_DIR/cpp/ $filename
done
