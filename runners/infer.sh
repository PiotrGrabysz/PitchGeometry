#!/bin/bash

IMAGE_NAME="pitch-geo-inference"

docker run \
  -it \
  --gpus all \
	-v $PWD:/workspace \
	--rm \
	$IMAGE_NAME "$@"