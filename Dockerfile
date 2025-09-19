# ---- Base: CUDA 12.1 runtime (driver는 호스트 제공) ----
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive TZ=Asia/Seoul \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
    TRANSFORMERS_CACHE=/root/.cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget git bzip2 ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# ---- Miniconda 설치 ----
ARG CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh && \
    bash /tmp/mc.sh -b -p $CONDA_DIR && \
    rm -f /tmp/mc.sh && \
    conda clean -afy

# Anaconda defaults 채널 TOS 동의(비대화식 빌드 용)
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# ---- Python 3.9 환경 생성 ----
RUN conda create -y -n remdm python=3.9 && \
    echo "conda activate remdm" >> /root/.bashrc
ENV PATH=$CONDA_DIR/envs/remdm/bin:$PATH

# ---- PyTorch 2.2.2 (CUDA 12.1 빌드) 설치 ----
RUN pip install --upgrade pip && \
    pip install --index-url https://download.pytorch.org/whl/cu121 \
        torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2

# ---- 프로젝트 의존성(토치 제외) ----
WORKDIR /workspace
COPY requirements-no-torch.txt .
RUN pip install -r requirements-no-torch.txt

# 기본 엔트리(검사용). VESSL에서는 Start Command에서 덮어씁니다.
CMD ["python", "--version"]
