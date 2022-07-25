# Ubuntu with desktop and VNC server
FROM consol/ubuntu-xfce-vnc

# Switche default user to root
USER 0

# Use bash for the shell
SHELL ["/bin/bash", "-c"]

# Set the environment so that we can use conda after install
ENV PATH='~/anaconda3/condabin:${PATH}'

# Used for GPU setup
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute, video, utility

# Needed to build some packages
RUN apt-get update \
&& apt-get upgrade -y \
&& apt-get install gcc -y \
&& apt-get install build-essential -y \
&& apt-get install unzip -y \
&& apt-get install nomacs -y \
&& apt-get install git -y \
&& apt-get install git -y \
&& apt-get install nano -y \
&& apt-get install wget -y \
&& apt-get install gedit -y \
&& apt-get install imagemagick -y \
&& apt-get install cmake -y 

# Get Anaconda package & install
RUN wget https:/repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh \
&& chmod 777 Anaconda3-2019.03-Linux-x86_64.sh \
&& ./Anaconda3-2019.03-Linux-x86_64.sh -b \
&& echo "export PATH=\"/headless/anaconda3/condabin:$PATH\"">>bashrc \
&& source ~/.bashrc

# Get CUDA & install
RUN wget https://developer.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda_10.1.105_418.39_linux.run \
&& chmod 777 cuda_10.1.105_418.39_linux.run

RUN ./cuda_10.1.105_418.39_linux.run --silent --toolkit

# Get CuDNN (put in current folder, download in https://developer.nvidia.com/rdp/cudnn-archive, need registration)
COPY ./cudnn.tgz ./cuda_10
RUN tar -xzvf cudnn.tgz \
&& cp cuda/lib64/libcudnn*.h /usr/local/include \
&& cp cuda/lib64/libcudnn* /usr/local/cuda/lib64 \
&& chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*

# Init PyTorch 1.9 (CUDA 9 compatible)
RUN git clone -b release/1.9 --recursive https://github.com/pytorch/pytorch \
&& cd pytorch \
&& git submodule sync \
&& git submodule update --init --recursive

# Get Torchvision (latest, otherwise is "0.10.0" for torch 1.9)
RUN git clone https://github.com/pytorch/vision.git

# Conda environment
RUN conda init \
&& conda create -n vid_sum python \
&& conda install -n vid_sum numpy ninja pyyaml mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses \
&& conda install -n vid_sum h5py \
&& conda install -n vid_sum tqdm \
&& conda install -n vid_sum pandas \
&& conda install -n vid_sum matplotlib \
&& conda install -n vid_sum opencv \
&& conda install -n vid_sum -c pytoch magma-cuda101

RUN conda init bash \
&& source ~/.bashrc \
&& conda activate vid_sum \
&& pip install ortools

# Dev Pytorch
RUN cd pytorch \
&& conda init bash \
&& source ~/.bashrc \
&& conda activate vid_sum \
&& python setup.py develop && python -c "import torch"

# Sup
RUN apt-get install software-properties-common -y \
&& add-apt-repository ppa:ubuntu-toolchain-r/test -y \
&& apt-get update -y \
&& apt-get install libstdc++6 -y

RUN apt-get install python3-setuptools -y

# Dev Torchvision
RUN cd ~/vision \
&& conda init bash \
&& source ~/.bashrc \
&& conda activate vid_sum \
&& python setup.py develop





