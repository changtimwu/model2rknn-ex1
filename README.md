# Model2RKNN

![python-3.x](https://img.shields.io/badge/Python-3.x-blue)

The repository targets the converting model from PyTorch /w ONNX format into the one of RKNN. 

## Overview

The flow of convertion can briefly break down into three steps, and each can be retrieved on the github.com as following.

### Step 1 

Edit the model, or train it if necessary, and export it as PyTorch-based (e.g. .pth, .pth.tar, etc.) or ONNX-based model (e.g. .onnx). You can find the relative operations from the below links.

- Resources:
  - [qpjkw/rknn-py](https://github.com/qpjkw/rknn-py)

### Step 2

Converting the model exported from the first step to RKNN by using the Python API. You can find the partial Python source code on the repository, or using the Python wheel from the [link](https://github.com/rockchip-linux/rknn-toolkit2/tree/master/packages) as well.

- Resources
  - [changtimwu/model2rknn-ex1](https://github.com/changtimwu/model2rknn-ex1.git)
  - [qpjkw/model2rknn-ex1](https://github.com/qpjkw/model2rknn-ex1)
  - [rockchip-linux/rknn-toolkit2](https://github.com/rockchip-linux/rknn-toolkit2/tree/master/packages)

You can dive into the `model2rknn-ex1` repoistory to the file `mynet/test.py` for more details. You can find the latest python wheel on the `rockchip-linux/rknn-toolkit2` repository.

### Step 3

Run the RKNN model on the Rockchip by using the binary executable. It was built on pure C++, cross compiled by the same aarch64 chipset, linked with official shared library on the same machine, and in the final moving to the Rockchip for executing.

- Resources
  - [qpjkw/rknpu2](https://github.com/qpjkw/rknpu2)

## How to Start

The following lists several approaches of converting model and testing on the x86 simulator.

### Python Virtual Environment

First, please prepare the python virtualenv for running the converting script.

```sh
# assume git cloning from qpjkw/model2rknn-ex1
cd ./model2rknn-ex1

python3 -V  # make sure the Python version is 3.6.x

python3 -m pip install --no-cache-dir virtualenv
python3 -m virtualenv -p python3 env

source ./env/bin/activate

python3 -m pip install --no-cache-dir -r ./usr/requirements.txt
python3 -m pip install --no-cache-dir ./usr/rknn_toolkit2-1.1.1_5c458c6-cp36-cp36m-linux_x86_64.whl
```

Second, edit the script `mynet/start-all.sh` for converting the PyTorch or ONNX model.

```sh
# ---------------------------------------------
# Please update the following part.
# You should edit the following the
# MODEL_NAME, TYPE, MODEL_FILE.
# ---------------------------------------------
MODEL_NAME=qnapaicore_v1-6
TYPE=onnx
MODEL_FILE=/home/rknn/data/$MODEL_NAME.$TYPE
# ---------------------------------------------
```

```sh
vim mynet/start-all.sh
```

### Docker COntainer Environment

You can pull image from the internal docker registry (172.17.34.124:10443).

```sh
docker logout

docker login -u username registry:10443

docker pull registry:10443/
```

Otherwise, it is easy to build a docker image from scratch, and run it.

```sh
cd ./model2rknn-ex1

docker build -t rknn:1.0.0 -f ./usr/rknn.dockerfile .

docker run -it --rm -v $PWD/tmp:/home/rknn/data rknn:1.0.0 bash
```

**Make sure you have mounted the volume under /home/rknn/data. A model named qnapaicore_v1-6.onnx was also found.** You now can run the shell script.

```sh
./start-all.sh
```
