FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu20.04

RUN apt-get update && apt-get install -y \
    curl \
    unzip \
    git \
    python3-pip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://bun.sh/install | bash

ENV PATH="/root/.bun/bin:${PATH}"

CMD ["sleep", "infinity"]