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

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# ...I'll just leave this here for now
RUN git clone https://github.com/xsuite/xobjects
RUN pip install -e xobjects

RUN git clone https://github.com/xsuite/xpart
RUN pip install -e xpart

RUN git clone https://github.com/xsuite/xtrack
RUN pip install -e xtrack

RUN git clone https://github.com/xsuite/xfields
RUN pip install -e xfields

COPY . /definitive_dyn_indicators
RUN pip install -e ./definitive_dyn_indicators

# create non-root user
RUN useradd -m -d /home/user -s /bin/bash user
RUN echo "user:user" | chpasswd
RUN mkdir -p /home/user/cupy_tmp
RUN chown -R user:user /home/user
# give user permission to write in /home/user
RUN chmod -R u+rwx /home/user
# give user permission to write in /home/user/cupy_tmp
RUN chmod -R u+rwx /home/user/cupy_tmp
# give everyone permission to write in /home/user/cupy_tmp
RUN chmod -R o+rwx /home/user/cupy_tmp
# give all permission to write in /home/user/cupy_tmp
RUN chmod -R 777 /home/user/cupy_tmp
RUN chmod -R 777 /home/user

# set environment variables
ENV CUPY_CACHE_DIR /home/user/cupy_tmp
# set user
USER user
# set working directory
WORKDIR /home/user