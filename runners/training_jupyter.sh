#!/bin/bash

IMAGE_NAME="pitch-geo-training-jupyter"

docker run \
  -it \
  --gpus all \
  --network host \
	-v "$PWD":/workspace \
	--rm \
	$IMAGE_NAME