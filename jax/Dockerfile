ARG CUDA="12.2.2"
ARG CUDNN="8"
ARG TAG="devel"
ARG OS="ubuntu20.04"
ARG JAX_CUDA_CUDNN="cuda12"
FROM nvidia/cuda:${CUDA}-cudnn${CUDNN}-${TAG}-${OS}

RUN apt-get update && \
    apt-get install -y \
        git \
        vim \
        htop \
        python3 \
        python3-pip && \
    rm -rf /var/lib/apt/lists/*

#ARG USER_ID
#ARG GROUP_ID
#ARG NAME
#RUN groupadd --gid ${GROUP_ID} ${NAME}
#RUN useradd \
    #--no-log-init \
    #--create-home \
    #--uid ${USER_ID} \
    #--gid ${GROUP_ID} \
    #-s /bin/sh ${NAME}

ARG WORKDIR_PATH
WORKDIR ${WORKDIR_PATH}

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install jax jaxlib[cuda]==0.4.12+cuda12.cudnn89 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

WORKDIR /app
COPY requirements.txt .
COPY jax_conv_lstm.py .
COPY rnn_config.yaml .

RUN python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu
RUN python3 -m pip install -r requirements.txt

CMD python3 jax_conv_lstm.py rnn_config.yaml
