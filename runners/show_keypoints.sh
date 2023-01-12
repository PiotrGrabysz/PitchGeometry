#!/bin/bash

IMAGE_NAME="pitch-geo-visualization"

docker run \
  -it \
	-v $PWD:/workspace \
	--rm \
	$IMAGE_NAME "$@"