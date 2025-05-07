FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN set -x && \
    echo "Acquire { HTTP { Proxy \"$HTTP_PROXY\"; }; };" | tee /etc/apt/apt.conf

ARG INSTALL_DIR=/opt
ARG BUILD_DIR=/app

WORKDIR $BUILD_DIR

COPY . $BUILD_DIR

ENV CUDA_PATH=/usr/local/cuda \
    TZ=Europe/Moscow \
    LC_CTYPE=en_US.UTF-8 \
    LANG=en_US.UTF-8

RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    wget unzip cmake git software-properties-common ninja-build && \
    rm -rf /var/lib/apt/lists/*

RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install --no-install-recommends -y \
    python3.12 \
    python3.12-dev && \
    rm -rf /var/lib/apt/lists/*

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3.12 get-pip.py && \
    rm get-pip.py

RUN ln -sf "$(which pip3.12)" /usr/local/bin/pip

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/local/bin/pip 1


RUN pip install --upgrade pip && \
    pip install -r requirements.txt
