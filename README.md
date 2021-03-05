-# Data Science Pragmatic Programming

  Snippet codes for common used routines in data science projects.

  * [python and pandas](ds_pragmatic_programming_python.ipynb)
  * [Big Data](ds_pragmatic_programming_pyspark.ipynb)
  * [SQL](ds_pragmatic_programming_SQL.ipynb)


  [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/leandroohf/ds_pragmatic_programming.git/master) <= **Take while when run for the first time** 

## Automated system
### Spark notebook

You need to run inside a docker container that has all packages and spark
running


https://hub.docker.com/r/ucsddse230/cse255-dse230


```sh
# Install images
docker pull ucsddse230/cse255-dse230

# Run container
docker run --name ds_pragmatic -it -p 8890:8888 -v /media/leandroohf/sdb1/leandro/ds_pragmatic_programming:/home/ucsddse230/ ucsddse230/cse255-dse230 /bin/bash

# Run jupyter inside container
jupyter notebook


# If you need to ssh to container
docker exec -it ds_pragmatic /bin/bash

```

1. http://localhost:8890/tree
2. Copy and paste token to login in the notebook


You do not have to, but in case you want to build the images

```sh
mkdir -p builddir
cd builddir

# !?
curl asdfa

docker build -t ucsddse230/cse255-dse230 .

# Run for testing
docker run -it -p 8888:8888 ucsddse230/cse255-dse230 jupyter notebook 
```

 Un compress wetaher data
 
 ```sh
 tar -zxvf data/Weather/NY.tgz -C data/Weather
 ```

### Deep Learning notebook

https://hub.docker.com/r/ermaker/keras-jupyter
https://github.com/ermaker/dockerfile-keras-jupyter


```sh
# Install images
docker pull bowen0701/docker-python3-ml-jupyter

# Run container
docker run --name deep_learning -it -p 8888:8888 -p 6006:6006 \
       -v ~/Documents/leandro/castingworkbook:/notebooks  \
       bowen0701/docker-python3-ml-jupyter

# If you need to ssh to container
docker exec -it deep_learning /bin/bash

```

1. http://localhost:8888/tree
2. Copy and paste token to login in the notebook


In case you want to build the image and add packages


1. Save the Dockerfile in a folder build

```txt
# Docker settings: Ubuntu, Python3, pip, general machine learning frameworks, Jupyter Notebook.

FROM ubuntu:16.04

LABEL maintainer="Bowen Li <bowen0701@gmail.com>"

# Pick up some Python3 dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        # python \
        # python-dev \
        # python-pip \
        # python-setuptools \
        python3 \
        python3-dev \
        python3-pip \
        python3-setuptools \
        rsync \
        software-properties-common \
        unzip \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python3 general packages.
RUN pip3 install --upgrade pip \
        numpy \
        scipy \
        pandas \
        sklearn \
        ipykernel \
        jupyter \
        notedown \
        matplotlib \
        seaborn \
        Cython \
        Pillow \
        requests \
        awscli \
        && \
    python3 -m ipykernel.kernelspec

# Install machine learning packages.
RUN pip3 --no-cache-dir install --upgrade \
        tensorflow \
        tensorboard==1.12 \
        mxnet \
        http://download.pytorch.org/whl/cpu/torch-0.3.1-cp35-cp35m-linux_x86_64.whl \
        torchvision \
        xgboost \
        pymc3 \
        pystan \
        gensim \
        nltk \
        opencv-python

# Set up our notebook config.
COPY jupyter_notebook_config.py /root/.jupyter/

# Copy sample notebooks.
COPY notebooks /notebooks

# Jupyter has issues with being run directly:
#   https://github.com/ipython/ipython/issues/7062
# We just add a little wrapper script.
COPY run_cmd.sh /

# Jupyter Notebook
EXPOSE 8888
# TensorBoard
EXPOSE 6006

WORKDIR /notebooks

RUN chmod +x /run_cmd.sh

CMD ["/run_cmd.sh"]
```
