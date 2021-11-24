# use latest version of cuda
FROM nvidia/cuda:11.4.2-devel-ubuntu20.04

# install packages for ubuntu
ENV TZ=Europe/Zurich
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && \
    echo $TZ > /etc/timezone && \
    apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    cmake \
    libfftw3-dev \
    python3 \
    python3-dev \
    python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# install requirements
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# set environment variables for cupy
ENV CUPY_CACHE_DIR /tmp/cupy_cache

# install my henon map package
RUN git clone --recurse-submodules https://github.com/carlidel/henon_map_cpp && \
    pip install -e ./henon_map_cpp

# ...I'll just leave this here for now
RUN git clone https://github.com/xsuite/xobjects && \
    pip install -e xobjects && \
    git clone https://github.com/xsuite/xpart && \
    pip install -e xpart && \
    git clone https://github.com/xsuite/xtrack && \
    pip install -e xtrack && \
    git clone https://github.com/xsuite/xfields && \
    pip install -e xfields

RUN mkdir /definitive_dyn_indicators

ADD masks /definitive_dyn_indicators/masks
COPY setup.py /definitive_dyn_indicators/setup.py
ADD definitive_dyn_indicators /definitive_dyn_indicators/definitive_dyn_indicators

RUN pip install -e ./definitive_dyn_indicators