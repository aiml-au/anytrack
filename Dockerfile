# cuda docker from pytroch
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Install pip and apt packages
COPY requirements.txt .
RUN python -m pip install -r requirements.txt

# required to run apt update without errors
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update 

# this is required for cv2 to work inside this docker
RUN apt-get install ffmpeg libsm6 libxext6  -y

# installing other apt packages 
RUN apt-get install -y wget 
RUN apt-get install -y \
        git \
        openssh-server \
        libmysqlclient-dev
RUN rm -rf /var/lib/apt/lists/*


WORKDIR /app
COPY . /app

# Setting up conda env
RUN conda env create -f environment_gpu.yml
RUN echo "source activate anytrack" > ~/.bashrc
RUN bash -c "source ${HOME}/.bashrc" 


