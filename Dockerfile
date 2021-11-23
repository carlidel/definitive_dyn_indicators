# use latest version of cuda
FROM nvidia/cuda:11.4.2-devel-ubuntu20.04

# install packages for ubuntu
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    python3 \
    python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY . /definitive_dyn_indicators

# install python packages
RUN pip install --no-cache-dir -r definitive_dyn_indicators/requirements.txt
RUN pip install -e ./definitive_dyn_indicators