#!/bin/bash

IMAGE_NAME="pitch-geo-training-dev"

docker run \
  -it \
  --gpus all \
	-v $PWD:/workspace \
	--rm \
	$IMAGE_NAME