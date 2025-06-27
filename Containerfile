FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04

# DEFINIÇÕES INICIAIS  #########################################################
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/.cargo/bin:/root/.bun/bin:${PATH}"

# INSTALA AS DEPENDÊNCIAS ESSENCIAIS ###########################################
RUN apt-get update                          \
 && apt-get install -y                      \
    curl                                    \
    unzip                                   \
    git                                     \
    ffmpeg                                  \
    build-essential                         \
    software-properties-common              

# REVISIT IT: Use deadsnakes PPA to install Python 3.11 ########################
RUN add-apt-repository ppa:deadsnakes/ppa   \
 && apt-get install -y                      \
    python3.10                              \
    python3.10-dev                          \
    python3.10-venv                         \
 && update-alternatives --install           \
      /usr/bin/python3 python3              \
      /usr/bin/python3.10 1

# CLEANUP APT CACHE ############################################################
RUN apt-get clean                           \
 && rm -rf /var/lib/apt/lists/*   

# INSTALA O COMPILADOR RUST ####################################################
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# INSTALA O BUN ################################################################
RUN curl -fsSL https://bun.sh/install | bash

# DEFINE O DIRETÓRIO DE TRABALHO ###############################################
WORKDIR /workspace

# COMANDO PARA MANTER O CONTAINER RODANDO ######################################
CMD ["sleep", "infinity"]