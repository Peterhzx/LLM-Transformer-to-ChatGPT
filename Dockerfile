# mind that you need Nvidia Container Toolkit to use cuda images. https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
# FROM nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04
# nvcr.io/nvidia/pytorch:25.03-py3

FROM nvcr.io/nvidia/pytorch:25.03-py3
LABEL authors="Peterhzx"

# update-alternatives: sets Python 3.10 as the default python3 in the container.
# RUN apt-get update && \
#     apt-get install -y python3 python3-pip && \
#     python3 -m pip install --upgrade pip && \
#     apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT ["python3"]

CMD train.py