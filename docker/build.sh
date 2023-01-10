#!/bin/bash

BASE_IMAGE_NAME="pitch-geo-tensorflow"

docker pull tensorflow/tensorflow:2.9.1-gpu
docker build -f docker/tensorflow.Dockerfile -t $BASE_IMAGE_NAME .
docker build -f docker/training.Dockerfile -t "pitch-geo-training" .
docker build -f docker/dev.Dockerfile -t "pitch-geo-training-dev" .
docker build -f docker/inference.Dockerfile -t "pitch-geo-inference" .
docker build -f docker/visualization.Dockerfile -t "pitch-geo-visualization" .