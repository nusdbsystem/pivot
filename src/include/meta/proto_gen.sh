#!/bin/bash
SRC_DIR=../meta
DST_DIR=../protobuf/
protoc -I=$SRC_DIR --cpp_out=$DST_DIR $SRC_DIR/*.proto