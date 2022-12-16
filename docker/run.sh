#!/bin/bash

IMAGE_NAME="pitch_geo"
TAG="v2"

docker run \
  -it \
  --gpus all \
	--network host \
	-v $PWD:/workspace \
	--rm \
	$IMAGE_NAME:$TAG