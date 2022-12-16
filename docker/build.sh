#!/bin/bash

IMAGE_NAME="pitch_geo"
TAG="v2"

docker pull tensorflow/tensorflow:2.9.1-gpu
docker build -f docker/Dockerfile -t $IMAGE_NAME:$TAG .