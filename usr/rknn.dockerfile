FROM ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y libssl-dev python3-pip && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip && \
    pip3 install --no-cache-dir Cython==0.29.22 numpy==1.19.5

WORKDIR /usr/rknn/usr/

COPY ./usr/rknn_toolkit2-1.1.1_5c458c6-cp36-cp36m-linux_x86_64.whl ./
COPY ./usr/requirements.txt ./

RUN pip install --no-cache-dir -r ./requirements.txt
RUN pip install --no-cache-dir ./rknn_toolkit2-1.1.1_5c458c6-cp36-cp36m-linux_x86_64.whl

RUN apt-get update && \
    apt-get install ffmpeg libsm6 libxext6 libprotobuf10 -y && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir opencv-python 

RUN useradd -ms /bin/bash rknn
USER rknn

WORKDIR /home/rknn
COPY ./mynet ./

CMD ["bash"]
