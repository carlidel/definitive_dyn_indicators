# use latest version of cuda
FROM nvidia/cuda:11.4.2-devel-ubuntu20.04

# install packages for ubuntu
ENV TZ=Europe/Rome
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    cmake \
    libfftw3-dev \
    python3 \
    python3-dev \
    python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# install my henon map package
RUN git clone --recurse-submodules https://github.com/carlidel/henon_map_cpp
RUN pip install -e ./henon_map_cpp

COPY . /definitive_dyn_indicators

# install python packages
RUN pip install --no-cache-dir -r definitive_dyn_indicators/requirements.txt
RUN pip install -e ./definitive_dyn_indicators