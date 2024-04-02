FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04

RUN apt-get update && apt-get install -y \
    git \
    curl \
    ca-certificates \
    python3 \
    python3-pip \ 
    sudo \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m username

RUN chown -R username:username /home/username

COPY --chown=username *.* /home/username/app

USER username
RUN mkdir /home/username/data/

RUN cd /home/username/app/ && pip3 install -r requirements.txt

RUN pip3 install mkl 

WORKDIR /home/username/app

# building the docker container
# docker build -f Dockerfile -t bert:train . #bert:train is the docker container name

# log into the container
# docker run -ti bert:train /bin/bash

# running model
# docker run -ti bert:train python3 train.py