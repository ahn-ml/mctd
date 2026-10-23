FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED 1
ENV PIP_DISABLE_PIP_VERSION_CHECK 1
ENV PIP_NO_CACHE_DIR 1
RUN apt-get update && apt-get install -y \
  ffmpeg git python3-pip vim libglew-dev cmake \
  x11-xserver-utils xvfb \
  && apt-get clean
RUN pip3 install --upgrade pip

# Install libraries
RUN pip3 install torch~=2.0.0
RUN pip3 install torchvision~=0.15.1
RUN pip3 install lightning~=2.1.2
RUN pip3 install wandb~=0.17.0
RUN pip3 install hydra-core~=1.3.2
RUN pip3 install omegaconf~=2.3.0
RUN pip3 install torchmetrics[image]==0.11.4
RUN pip3 install wandb-osh==1.2.1
RUN pip3 install gluonts[torch]==0.13.1
RUN pip3 install pytorchvideo~=0.1.5
RUN pip3 install colorama
RUN pip3 install tqdm
RUN pip3 install opencv-python
RUN pip3 install matplotlib
RUN pip3 install click
RUN pip3 install moviepy
RUN pip3 install imageio
RUN pip3 install einops
RUN pip3 install pandas
RUN pip3 install pyzmq
RUN pip3 install pyrealsense2
RUN pip3 install internetarchive

# From extra requirements
RUN pip3 install cython==0.29.37
RUN pip3 install setuptools==65.5.0
RUN pip3 install wheel==0.40.0
RUN pip3 install d4rl==1.1
RUN pip3 install shimmy>=0.2.1
RUN pip3 install stable-baselines3
RUN pip3 install pip==21.0.1
RUN pip3 install gym==0.19.0
RUN pip3 install mujoco-py
RUN pip3 install rotary-embedding-torch
RUN pip3 install diffusers

# Install additional libraries
RUN apt-get update
RUN apt-get -y install libosmesa6-dev patchelf

# Upgrade torchvision to remove warning msgs
RUN pip3 install --upgrade torchvision

# Copy the mujoco folder
COPY ./mujoco/generated /usr/local/lib/python3.10/dist-packages/mujoco_py/generated

# Give the authority to the user for the mujoco_py library
RUN chmod -R 777 /usr/local/lib/python3.10/dist-packages/mujoco_py/generated

# Install additional libraries for OGBench (the bundled wheel is required; stock OGBench differs)
ENV MUJOCO_GL=egl
RUN apt install -y mesa-utils
COPY ./ogbench-1.0.1-py3-none-any.whl .
RUN pip3 install ./ogbench-1.0.1-py3-none-any.whl
RUN pip3 install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
RUN pip3 install moviepy==1.0.3 imageio-ffmpeg

# Graph construction (k-means over dataset states)
RUN pip3 install scikit-learn

# New user; match the host user to keep mounted files writable
ARG UNAME=user
ARG UID=1000
ARG GID=1000
RUN groupadd -g $GID -o $UNAME
RUN useradd -m -u $UID -g $GID -o -s /bin/bash $UNAME
USER $UNAME
WORKDIR /home/$UNAME

# Copy the mujoco folder
RUN mkdir -p /home/$UNAME/.mujoco/mujoco210
COPY ./mujoco/mujoco210 /home/$UNAME/.mujoco/mujoco210

# Add LD_LIBRARY_PATH
ENV MUJOCO_PY_MUJOCO_PATH /home/$UNAME/.mujoco/mujoco210
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64:/home/$UNAME/.mujoco/mujoco210/bin
