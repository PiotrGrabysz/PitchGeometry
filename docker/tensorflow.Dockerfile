FROM tensorflow/tensorflow:2.9.1-gpu

ENV TERM=xterm
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/3bf863cc.pub

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libopencv-dev \
    libsm6 \
    libxext6 \
    python3-opencv \
    python3-pil \
    python3-lxml \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

ARG uid=1000
ARG gid=1000
RUN groupadd -r -f -g ${gid} tensorflow-user && useradd -o -r -l -u ${uid} -g ${gid} -ms /bin/bash tensorflow-user
RUN usermod -aG sudo tensorflow-user
RUN echo 'tensorflow-user:tensorflow' | chpasswd
RUN mkdir -p /workspace && chown tensorflow-user /workspace -R

USER tensorflow-user
ENV PATH="/home/tensorflow-user/.local/bin:${PATH}"
RUN python3 -m pip install -U pip

ENV TF_CPP_MIN_LOG_LEVEL=3
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
ENV PYTHONPATH="/workspace"