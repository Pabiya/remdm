FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y wget git bzip2 && rm -rf /var/lib/apt/lists/*
# Miniconda
ARG CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh \
 && bash /tmp/mc.sh -b -p $CONDA_DIR && rm -f /tmp/mc.sh
RUN conda create -y -n remdm python=3.9 && echo "conda activate remdm" >> /root/.bashrc
ENV PATH=$CONDA_DIR/envs/remdm/bin:$PATH
RUN pip install --upgrade pip \
 && pip install --index-url https://download.pytorch.org/whl/cu121 \
      torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2
WORKDIR /workspace
COPY requirements-no-torch.txt .
RUN pip install -r requirements-no-torch.txt
