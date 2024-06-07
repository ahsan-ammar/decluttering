FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND noninteractive
ENV PYTHONUNBUFFERED True
ENV PORT 8080

ENV CUDA_HOME=/usr/local/cuda \
     TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX" \
     SETUPTOOLS_USE_DISTUTILS=stdlib

RUN conda update conda -y

# Install libraries in the brand new image. 
RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         unzip \
         build-essential \
         nginx \
         git \
         vim \
         python3-opencv \
         ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Return to the main directory
WORKDIR /usr/src/app

RUN apt-get update && apt-get install -y libgl1-mesa-glx
RUN apt-get update && apt-get install -y libglvnd0
RUN apt-get update && apt-get install -y xvfb

# Install requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model files
COPY . .

CMD python -u handler.py
# RUN python main.py
